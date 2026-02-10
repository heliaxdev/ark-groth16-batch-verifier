#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::iter;
use core::ops::Neg;

use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, VariableBaseMSM};
use ark_ff::fields::{Field, One, Zero};
use ark_groth16::data_structures::{PreparedVerifyingKey, Proof};
use ark_relations::r1cs::Result;
use ark_relations::r1cs::SynthesisError;
use ark_transcript::{IsLabel, Transcript};

#[repr(C)]
struct HeadTail<const N: usize, T> {
    head: T,
    tail: [T; N],
}

impl<const N: usize, T> HeadTail<N, T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        // SAFETY: (T, [T; N]) should have the same ABI as [T; N+1]
        unsafe { core::slice::from_raw_parts(&self.head as *const _ as *const _, N + 1) }
    }
}

/// VerifyingKey optimized for batch verification.
#[derive(Clone, Debug)]
pub struct BatchPreparedVerifyingKey<E: Pairing> {
    /// The alpha_g1 element
    pub alpha_g1: E::G1Affine,
    /// -beta_g2, prepared for the Miller loop
    pub neg_beta_g2_prepared: E::G2Prepared,
    /// -gamma_g2, prepared for the Miller loop
    pub neg_gamma_g2_prepared: E::G2Prepared,
    /// -delta_g2, prepared for the Miller loop
    pub neg_delta_g2_prepared: E::G2Prepared,
    /// The public input commitments (S_i)
    pub gamma_abc_g1: Vec<E::G1Affine>,
}

impl<E: Pairing> BatchPreparedVerifyingKey<E> {
    /// Further processes a [`PreparedVerifyingKey`] for batch verification of Groth16
    /// proofs belonging to the same circuit.
    ///
    /// ## Implementation details
    ///
    /// This performs the expensive G2 negation and preparation step for beta
    /// upfront, saving time during repeated batch verifications.
    #[inline]
    pub fn from_pvk(pvk: &PreparedVerifyingKey<E>) -> Self
    where
        E::G2Affine: Neg<Output = E::G2Affine>,
    {
        Self {
            gamma_abc_g1: pvk.vk.gamma_abc_g1.clone(),
            alpha_g1: pvk.vk.alpha_g1,
            neg_beta_g2_prepared: E::G2Prepared::from(-pvk.vk.beta_g2),
            neg_gamma_g2_prepared: pvk.gamma_g2_neg_pc.clone(),
            neg_delta_g2_prepared: pvk.delta_g2_neg_pc.clone(),
        }
    }
}

/// Verify a batch of Groth16 proofs of the same circuit.
pub fn batch_verify<const BATCH_SIZE: usize, const NUM_PUB_INPUTS: usize, E>(
    bpvk: &BatchPreparedVerifyingKey<E>,
    proofs: &[Proof<E>; BATCH_SIZE],
    inputs: &[[E::ScalarField; NUM_PUB_INPUTS]; BATCH_SIZE],
    transcript: &mut Transcript,
) -> Result<bool>
where
    E: Pairing,
    E::G2Affine: Neg<Output = E::G2Affine>,
{
    if bpvk.gamma_abc_g1.len() != NUM_PUB_INPUTS + 1 {
        return Err(SynthesisError::MalformedVerifyingKey);
    }

    let challenges: [E::ScalarField; BATCH_SIZE] = compute_challenges(proofs, inputs, transcript);

    let challenges_sum = challenges
        .iter()
        .fold(E::ScalarField::zero(), |accum, r| accum + *r);

    let l_agg = {
        // the first scalar is for the constant '1' wire (sum of r_i)
        let head = challenges_sum;

        // compute the weighted sum for each public input column
        let tail = {
            let mut tail = [E::ScalarField::zero(); NUM_PUB_INPUTS];

            for i in 0..NUM_PUB_INPUTS {
                let mut sum = E::ScalarField::zero();
                for b in 0..BATCH_SIZE {
                    sum += challenges[b] * inputs[b][i];
                }
                tail[i] = sum;
            }

            tail
        };

        // NB: we use a hack to get a slice with NUM_PUB_INPUTS+1 elements,
        // since there are no const generic ops yet to statically initialize
        // an array with NUM_PUB_INPUTS+1 elements
        let input_scalars = HeadTail { head, tail };

        <E::G1 as VariableBaseMSM>::msm(&bpvk.gamma_abc_g1, input_scalars.as_slice())
            .map_err(|_| SynthesisError::Unsatisfiable)?
    };

    let c_agg = {
        let c_points = {
            let mut c_points = [E::G1Affine::zero(); BATCH_SIZE];

            for b in 0..BATCH_SIZE {
                c_points[b] = proofs[b].c;
            }

            c_points
        };
        <E::G1 as VariableBaseMSM>::msm(&c_points, &challenges)
            .map_err(|_| SynthesisError::Unsatisfiable)?
    };

    let g1_terms = {
        let p_alpha = bpvk.alpha_g1 * challenges_sum;
        let g1_proof_terms = (0..BATCH_SIZE).map(|b| proofs[b].a * challenges[b]);
        let g1_static_terms = iter::once(p_alpha)
            .chain(iter::once(c_agg))
            .chain(iter::once(l_agg));
        g1_proof_terms.chain(g1_static_terms)
    };
    let g2_terms = {
        let g2_proof_terms = proofs.iter().map(|p| E::G2Prepared::from(p.b));
        let g2_static_terms = iter::once(bpvk.neg_beta_g2_prepared.clone())
            .chain(iter::once(bpvk.neg_delta_g2_prepared.clone()))
            .chain(iter::once(bpvk.neg_gamma_g2_prepared.clone()));
        g2_proof_terms.chain(g2_static_terms)
    };

    // $$
    // \left( \prod_{i=1}^n e(r_i A_i, B_i) \right) \cdot \
    // e(R_{sum} \cdot \
    // \alpha, -\beta) \cdot \
    // e(C_{agg}, -\delta) \cdot \
    // e(L_{agg}, -\gamma) = 1_{\mathbb{G}_T}
    // $$
    //
    // rendered: https://quicklatex.com/cache3/9d/ql_873db65b459e222333fd68c1faf98f9d_l3.png
    let mml = E::multi_miller_loop(g1_terms, g2_terms);
    let result = E::final_exponentiation(mml).ok_or(SynthesisError::UnexpectedIdentity)?;

    Ok(result.0.is_one())
}

fn compute_challenges<const BATCH_SIZE: usize, const NUM_PUB_INPUTS: usize, E>(
    proofs: &[Proof<E>; BATCH_SIZE],
    inputs: &[[E::ScalarField; NUM_PUB_INPUTS]; BATCH_SIZE],
    transcript: &mut Transcript,
) -> [E::ScalarField; BATCH_SIZE]
where
    E: Pairing,
{
    // commit to all proofs and input
    transcript.label(b"batch_verify inputs");

    for b in 0..BATCH_SIZE {
        transcript.label(b"proof");
        transcript.append(&proofs[b].a);
        transcript.append(&proofs[b].b);
        transcript.append(&proofs[b].c);

        transcript.label(b"public inputs");
        for input in &inputs[b] {
            transcript.append(input);
        }
    }

    // generate random scalars. challenges[0] is set to 1, as per
    // the snarkpack paper optimization
    transcript.label(b"batch_verify challenges");

    let mut challenges = [const { E::ScalarField::ONE }; BATCH_SIZE];

    for (b, challenge) in challenges.iter_mut().enumerate().skip(1) {
        *challenge = {
            let mut reader = transcript.challenge(IsLabel({
                let mut buf = [b'r', b'a', b'n', b'd', 0, 0, 0, 0];
                let n = (b as u32).to_le_bytes();
                buf[4..].copy_from_slice(&n);
                buf
            }));

            reader.read_reduce()
        };
    }

    challenges
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ff::UniformRand;
    use ark_groth16::Groth16;
    use ark_relations::r1cs::{
        ConstraintSynthesizer, ConstraintSystemRef, LinearCombination, SynthesisError,
    };
    use ark_snark::SNARK;
    use ark_std::rand::{SeedableRng, rngs::StdRng};

    use super::*;

    #[derive(Clone)]
    struct DummyCircuit {
        pub a: Option<Fr>,
        pub b: Option<Fr>,
    }

    impl ConstraintSynthesizer<Fr> for DummyCircuit {
        fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<()> {
            let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
            let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
            let c = cs.new_input_variable(|| {
                let a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
                let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;
                Ok(a * b)
            })?;

            cs.enforce_constraint(
                LinearCombination::from(a),
                LinearCombination::from(b),
                LinearCombination::from(c),
            )?;

            Ok(())
        }
    }

    #[test]
    fn test_batch_verification() {
        let mut rng = StdRng::seed_from_u64(12345);

        let circuit_setup = DummyCircuit { a: None, b: None };
        let (pk, vk) =
            Groth16::<Bls12_381>::circuit_specific_setup(circuit_setup, &mut rng).unwrap();
        let bpvk = {
            let pvk = Groth16::<Bls12_381>::process_vk(&vk).unwrap();
            BatchPreparedVerifyingKey::from_pvk(&pvk)
        };

        const BATCH_SIZE: usize = 8;
        const NUM_PUB_INPUTS: usize = 1;

        let mut proofs_vec = Vec::with_capacity(BATCH_SIZE);
        let mut inputs_vec = Vec::with_capacity(BATCH_SIZE);

        for _ in 0..BATCH_SIZE {
            let a = Fr::rand(&mut rng);
            let b = Fr::rand(&mut rng);
            let c = a * b;

            let circuit = DummyCircuit {
                a: Some(a),
                b: Some(b),
            };

            let proof = Groth16::<Bls12_381>::prove(&pk, circuit, &mut rng).unwrap();

            proofs_vec.push(proof);
            inputs_vec.push([c]);
        }

        let proofs_array: [Proof<Bls12_381>; BATCH_SIZE] = proofs_vec
            .try_into()
            .unwrap_or_else(|_| panic!("Vec length mismatch"));

        let inputs_array: [[Fr; NUM_PUB_INPUTS]; BATCH_SIZE] = inputs_vec
            .try_into()
            .unwrap_or_else(|_| panic!("Vec length mismatch"));

        let mut transcript = Transcript::new_blank();

        let result = batch_verify::<BATCH_SIZE, NUM_PUB_INPUTS, Bls12_381>(
            &bpvk,
            &proofs_array,
            &inputs_array,
            &mut transcript,
        )
        .unwrap();

        assert!(result, "Batch verification failed!");
    }
}

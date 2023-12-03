use core::iter::zip;
use pulp::f64x4;
use pulp::x86::V3;

#[allow(non_camel_case_types)]
pub type c64x8x8 = [[[f64; 2]; 8]; 8];

// we use the same matmul strategy from gemm
// which consists of splitting the c64 mul_add into two f64 mul_adds (with alternating signs)
// given c64 inputs and outputs represented as f64x2
//
// c_re <- a_re * b_re - a_im * b_im + c_re
// c_im <- a_im * b_re - a_re * b_im + c_im
//
// this can be rewritten as
//
// c_re <- -(a_im * b_im - (a_re * b_re - c_re))
// c_im <-   a_re * b_im + (a_im * b_re + c_im)
//
// which can be simdified to
//
// NOTE: what pulp calls mul_subadd is what intel calls addsub
// tmp <- mul_subadd(a_swap, f64x2(b_im, b_im), mul_subadd(a, f64x2(b_re, b_re), c))
// c.0 <- -tmp.0
// c.1 <-  tmp.1

// 8 rows × 8 cols of a
// 2 cols × 8 rows of b
// 8 rows × 2 cols of c
// we can write a generic version depending on the number of rows and columns, instead of just 8x8
// but that would take some extra work
// for example, for 9x9, the columns don't fit neatly into registers, so we have to use partial
// loads and stores, or store the matrix into 10x9 storage
//
// the generic version is gonna be faster than the gemm version for small matrices.
// once the matrices no longer fit in the cache, we're better off calling the gemm implementation
// we're already going to L1 regularly
// for every call to kernel_8x2x8_avx, we reload a from memory
// for example here we could change 8 (a.nrows) into some const generic M: usize
// we'd want a specialization for values of M that are multiples of the register size
// although if the column doesn't fit in registers, we'd have to split it into two (or more) matmuls
//
// small matmul has a lot more possible optimizations that the large one, so the number of
// possibilities grows very fast
// that's part of the reason why i haven't been able to dedicate time to fixing it
#[inline(always)]
fn kernel_8x2x8_avx(
    simd: V3,
    c: &mut [[[f64; 2]; 8]; 2],
    a: &[[[f64; 2]; 8]; 8],
    b: &[[[f64; 2]; 8]; 2],
) {
    // standard registers only hold 64bits
    // we use the accumulator registers during the whole duration of the matmul
    // yeah
    let c = bytemuck::cast_mut::<_, [[f64x4; 4]; 2]>(c);
    let a = bytemuck::cast_ref::<_, [[f64x4; 4]; 8]>(a);

    // 8 registers
    let mut acc = [[simd.splat_f64x4(0.0); 4]; 2];
    // split c and b into their 2 columns
    let [acc0, acc1] = &mut acc;
    let [b0, b1] = b;

    // gemm does some extra work, first to store a and b in aligned storage
    // then it has to do all the loops dynamically since it doesn't know how many iterations there
    // are
    // yeah
    // although, it might also work for 8×k times k×8
    for (a_k, (bk0, bk1)) in zip(a, zip(b0, b1)) {
        // 4 registers for a
        // 2 registers for b

        // so, the components of a are loaded into the same register
        // the components of b are loaded separately
        // so it's a bit hybrid
        // that's what we have the simd instructions for

        // re im re im
        let a_k = *a_k;
        let bk0_re = simd.splat_f64x4(bk0[0]);
        let bk1_re = simd.splat_f64x4(bk1[0]);

        for ((acc0, acc1), a) in zip(zip(&mut *acc0, &mut *acc1), a_k) {
            *acc0 = simd.mul_subadd_f64x4(a, bk0_re, *acc0);
            *acc1 = simd.mul_subadd_f64x4(a, bk1_re, *acc1);
        }

        // this swaps the real and imaginary components of a
        // im re im re
        let a_k_swap = [
            pulp::cast::<_, f64x4>(simd.avx._mm256_permute_pd::<0b0101>(pulp::cast(a_k[0]))),
            pulp::cast::<_, f64x4>(simd.avx._mm256_permute_pd::<0b0101>(pulp::cast(a_k[1]))),
            pulp::cast::<_, f64x4>(simd.avx._mm256_permute_pd::<0b0101>(pulp::cast(a_k[2]))),
            pulp::cast::<_, f64x4>(simd.avx._mm256_permute_pd::<0b0101>(pulp::cast(a_k[3]))),
        ];

        // it's fused multiply add or sub
        // the first element in the simd lane gets subtraction
        // the second gets addition
        // third gets subtraction
        // 4th, addition
        // yeah, specialized for complex mul
        let bk0_im = simd.splat_f64x4(bk0[1]);
        let bk1_im = simd.splat_f64x4(bk1[1]);
        for ((acc0, acc1), a) in zip(zip(&mut *acc0, &mut *acc1), a_k_swap) {
            *acc0 = simd.mul_subadd_f64x4(a, bk0_im, *acc0);
            *acc1 = simd.mul_subadd_f64x4(a, bk1_im, *acc1);
        }
    }

    // now we store the result into c
    // but before that, we need to fix the signs of the accumulator
    // in the above formula, acc now holds `tmp`, which is -conj(final_result)
    // so we need to apply a negative sign to the real part to fix it
    // 0.0  has the bit pattern 0000...
    // -0.0 has the bit pattern 1000...
    let sign = pulp::cast::<_, f64x4>([-0.0, 0.0, -0.0, 0.0]);
    // we xor by sign to apply the right signs

    for acc0 in &mut *acc0 {
        *acc0 = simd.xor_f64x4(sign, *acc0);
    }
    for acc1 in &mut *acc1 {
        *acc1 = simd.xor_f64x4(sign, *acc1);
    }

    // now we store into c
    c[0] = *acc0;
    c[1] = *acc1;

    // hopefully we didn't mess up somewhere
}

pub fn matmul_8x8_avx(simd: V3, c: &mut c64x8x8, a: &c64x8x8, b: &c64x8x8) {
    struct Impl<'a> {
        simd: V3,
        c: &'a mut c64x8x8,
        a: &'a c64x8x8,
        b: &'a c64x8x8,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self { simd, c, a, b } = self;
            let c = bytemuck::cast_mut::<c64x8x8, [[[[f64; 2]; 8]; 2]; 4]>(c);
            let b = bytemuck::cast_ref::<c64x8x8, [[[[f64; 2]; 8]; 2]; 4]>(b);

            // 1 column = 8×c64 = 16×f64 = 4×(4×f64)
            // 1 register = 4×f64

            // the idea is that we want to compute the matmul like this
            //
            // c[:, j] = sum a[:, k] * b[k, j]
            //
            // this works nicely for us because each column of c (and a) fits nicely into registers
            // assuming avx2 + fma, (16 registers total) we get 4 registers per column
            //
            // so we need 4 registers for one column of a and 1 register to hold the value of b
            // currently being used
            //
            // this leaves us with 16 - 5 = 11 registers. since we have 4 registers per column,
            // we can work with 11(registers) / 4(registers per col) = 2 columns at a time (so
            // close to being 3, darn)
            //
            // so we're gonna be using 4 8x2 kernels
            //
            // well, across only one register, because we only need one value of b for the full
            // column of a

            kernel_8x2x8_avx(simd, &mut c[0], a, &b[0]);
            kernel_8x2x8_avx(simd, &mut c[1], a, &b[1]);
            kernel_8x2x8_avx(simd, &mut c[2], a, &b[2]);
            kernel_8x2x8_avx(simd, &mut c[3], a, &b[3]);
        }
    }

    simd.vectorize(Impl { simd, c, a, b });
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::array;
    use faer::complex_native::c64;
    use rand::random;

    #[test]
    fn test_matmul_8x8() {
        let a: c64x8x8 = array::from_fn(|_| array::from_fn(|_| [random(), random()]));
        let b: c64x8x8 = array::from_fn(|_| array::from_fn(|_| [random(), random()]));

        let a_faer = faer::Mat::from_fn(8, 8, |i, j| c64::new(a[j][i][0], a[j][i][1]));
        let b_faer = faer::Mat::from_fn(8, 8, |i, j| c64::new(b[j][i][0], b[j][i][1]));

        // i started with python when i was 18, so like 8 years ago
        // then started c++ during an internship in my 3rd year of uni, about 4 years ago
        // then rust 2 years ago
        // V3 is avx + avx2 + fma + everything before them
        // V4 is avx512f + avx512bw + ... + everything before them, including V3
        // we check V4 first, then V3, if neither is available we call the scalar impl
        let mut c: c64x8x8 = [[[0.0; 2]; 8]; 8];
        if let Some(simd) = V3::try_new() {
            matmul_8x8_avx(simd, &mut c, &a, &b);
        }

        let c_faer = faer::Mat::from_fn(8, 8, |i, j| c64::new(c[j][i][0], c[j][i][1]));
        let diff = &c_faer - &a_faer * &b_faer;
        // i guess cause im using the same algorithm as the one in gemm, just without all the extra
        // overhead from runtime loops
        assert!(diff.norm_max() <= 1e-10);
    }
}

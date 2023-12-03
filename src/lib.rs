use pulp::f64x4;
use pulp::x86::V3;
use pulp::Simd;

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
// tmp <- mul_subadd(a_swap, f64x2(b_im, b_im), mul_subadd(a, f64x2(b_re, b_re), c))
// c.0 <- -tmp.0
// c.1 <-  tmp.1

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
            let c = bytemuck::cast_mut::<c64x8x8, [[f64x4; 4]; 8]>(c);
            let a = bytemuck::cast_ref::<c64x8x8, [[f64x4; 4]; 8]>(a);
            let b = bytemuck::cast_ref::<c64x8x8, [[f64x4; 4]; 8]>(b);
        }
    }

    simd.vectorize(Impl { simd, c, a, b });
}

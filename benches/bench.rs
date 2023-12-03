use criterion::*;

fn bench_faer(criterion: &mut Criterion) {
    let one = faer::complex_native::c64::new(1.0, 0.0);

    let mut c = faer::Mat::from_fn(8, 8, |_, _| one);
    let a = faer::Mat::from_fn(8, 8, |_, _| one);
    let b = faer::Mat::from_fn(8, 8, |_, _| one);

    criterion.bench_function("bench-faer", |bencher| {
        bencher.iter(|| {
            faer_core::mul::matmul(
                c.as_mut(),
                a.as_ref(),
                b.as_ref(),
                None,
                one,
                faer_core::Parallelism::None,
            );
        })
    });
}

fn bench_ndarray(criterion: &mut Criterion) {
    let one = num_complex::Complex64::new(1.0, 0.0);

    let mut c = ndarray::Array2::from_shape_fn((8, 8), |(_, _)| one);
    let a = ndarray::Array2::from_shape_fn((8, 8), |(_, _)| one);
    let b = ndarray::Array2::from_shape_fn((8, 8), |(_, _)| one);

    criterion.bench_function("bench-ndarray", |bencher| {
        bencher.iter(|| {
            c = a.dot(&b);
        })
    });
}

fn bench_small_gemm(criterion: &mut Criterion) {
    use small_gemm_tutorial::*;
    let one = [1.0, 0.0f64];

    let mut c = [[one; 8]; 8];
    let a = [[one; 8]; 8];
    let b = [[one; 8]; 8];

    if let Some(simd) = pulp::x86::V3::try_new() {
        criterion.bench_function("bench-small-gemm", |bencher| {
            bencher.iter(|| {
                matmul_8x8_avx(simd, &mut c, &a, &b);
            })
        });
    }
}

criterion_group!(benches, bench_faer, bench_ndarray, bench_small_gemm);
criterion_main!(benches);

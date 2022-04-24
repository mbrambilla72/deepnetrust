use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[inline]
fn my_tanh(x: f64) -> f64 {
    if x < -20.0 {
        -1.0
    }
    // approximation is correct to 30 decimals
    else if x > 20.0 {
        1.0
    } else {
        x.tanh()
    }
}

#[inline]
fn tanh_levien(x: f64) -> f64 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;

    let a = x + (0.16489087 * x3) + (0.00985468 * x5);

    a / (1.0 + (a * a)).sqrt()
}

fn bench_my_tanh(n: u64) {
    // Optionally include some setup
    let x: f64 = 1.0001;

    for i in 1..n {
        black_box(my_tanh(x));
    }
}

fn bench_tanh_levien(n: u64) {
    // Optionally include some setup
    let x: f64 = 1.0001;

    for i in 1..n {
        black_box(tanh_levien(x));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("my_tanh", |b| b.iter(|| bench_my_tanh(black_box(100000))));
    c.bench_function("tanh_levien", |b| {
        b.iter(|| bench_tanh_levien(black_box(100000)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

use onnxruntime_test::deepfm::ONNXRunner;

use rand::Rng;

use ndarray::{array, Array};
use ort::OrtResult;

fn main() -> OrtResult<()> {
    let mut runner = ONNXRunner::new("deepfm.onnx")?;
    let input_shape: Vec<usize> = vec![10000, 92];
    let input_data = array![
        8.0000_f64,
        10.000_f64,
        1.0000_f64,
        1.0000_f64,
        4.0000_f64,
        2.0000_f64,
        3.0000_f64,
        1.0000_f64,
        2.0000_f64,
        7.0000_f64,
        3.0000_f64,
        2.0000_f64,
        2.0000_f64,
        5.0000_f64,
        17.000_f64,
        1.0000_f64,
        3.0000_f64,
        12.000_f64,
        15.000_f64,
        1.0000_f64,
        8.0000_f64,
        13.000_f64,
        2.0000_f64,
        1.0000_f64,
        4.0000_f64,
        18.000_f64,
        4.0000_f64,
        15.000_f64,
        8.0000_f64,
        11.000_f64,
        14.000_f64,
        7.0000_f64,
        4.0000_f64,
        9.0000_f64,
        2.0000_f64,
        11.000_f64,
        2.0000_f64,
        10.000_f64,
        7.0000_f64,
        20.000_f64,
        2.0000_f64,
        2.0000_f64,
        7.0000_f64,
        5.0000_f64,
        14.000_f64,
        12.000_f64,
        12.000_f64,
        8.0000_f64,
        1.0000_f64,
        8.0000_f64,
        10.000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        2.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        2.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        1.0000_f64,
        342000.0_f64,
        1.0000_f64,
        2.0000_f64,
        1.0000_f64,
        4.0000_f64,
        2.0000_f64,
        1.0000_f64,
        375.00_f64,
        72.000_f64,
        79.000_f64,
        8.0000_f64,
        411.00_f64,
        79.000_f64,
        80.000_f64,
        2.0000_f64,
    ];
    let mut rng = rand::thread_rng();

    let mut input = Array::<f64, _>::zeros((input_shape[0], input_shape[1]));
    for i in 0..input_shape[0] {
        for j in 0..input_shape[1] {
            input[[i, j]] = if input_data[j] != 0.0_f64 {
                rng.gen::<f64>() * input_data[j]
            } else {
                0.0_f64
            }
        }
    }

    runner.run(input).unwrap();

    Ok(())
}

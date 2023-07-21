use rand::Rng;
use std::{path::Path, sync::Arc};

use ndarray::{array, Array, ArrayBase};
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, Session, SessionBuilder,
};

fn argmax_1d(
    array: &ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>,
) -> Option<(usize, f32)> {
    if array.is_empty() {
        return None;
    }

    let mut max_index = 0;
    let mut max_value = -std::f32::INFINITY;

    for (i, &value) in array.iter().enumerate() {
        if value > max_value {
            max_index = i;
            max_value = value;
        }
    }

    Some((max_index, max_value))
}

pub struct ONNXRunner {
    session: Session,
}

impl ONNXRunner {
    pub fn new<P: AsRef<Path>>(path: P) -> OrtResult<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("DeepFM")
                .with_execution_providers([ExecutionProvider::cuda()])
                .build()?,
        );

        // onnx の読み込み
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(3)?
            .with_model_from_file(path)?;

        Ok(Self { session })
    }

    pub fn run(&mut self, input: Array<f64, ndarray::Dim<[usize; 2]>>) -> anyhow::Result<()> {
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = self
            .session
            .run([InputTensor::from_array(input.into_dyn())])?;

        let output: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let output_view = output.view().to_owned();

        let (_argmax, _max_value) = argmax_1d(&output_view).unwrap();
        println!("{}, {}", _argmax, _max_value);
        Ok(())
    }
}

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

use std::{path::Path, sync::Arc};

use ndarray::{Array, ArrayBase};
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, Session, SessionBuilder,
};

pub fn argmax_1d(
    array: &ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>,
) -> Option<(usize, f32)> {
    if array.is_empty() {
        return None;
    }

    let mut max_index = 0;
    let mut max_value = -std::f32::INFINITY;

    for (i, &value) in array.iter().enumerate() {
        // println!("{}: {}", i, value);
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

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_model_from_file(path)?;

        Ok(Self { session })
    }

    pub fn run(&mut self, input: Array<f64, ndarray::Dim<[usize; 2]>>) -> anyhow::Result<()> {
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = self
            .session
            .run([InputTensor::from_array(input.into_dyn())])?;

        let output: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let output_view = output.view().to_owned();

        let (argmax, max_value) = argmax_1d(&output_view).unwrap();
        println!("{}, {}", argmax, max_value);
        Ok(())
    }
}

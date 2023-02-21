use std::sync::Arc;

use ndarray::{Array4, ArrayBase};
use ort::{
    download::vision::ImageClassification::MobileNet,
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder,
};

fn argmax_1d(
    array: &ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>,
) -> Option<(usize, f32)> {
    if array.is_empty() {
        return None;
    }

    let mut max_index = 0;
    let mut max_value = -std::f32::INFINITY;
    let mut sum_value = 0.0;

    for (i, &value) in array.iter().enumerate() {
        sum_value += value.exp();
        if value > max_value {
            // println!("{}: {}", i, value);
            max_index = i;
            max_value = value;
        }
    }

    // ついでに softmax している
    Some((max_index, max_value.exp() / sum_value))
}

fn main() -> OrtResult<()> {
    let environment = Arc::new(
        Environment::builder()
            .with_name("MobileNet")
            .with_execution_providers([ExecutionProvider::cuda()])
            .build()?,
    );

    // onnx の読み込み
    // https://github.com/onnx/models/tree/main/vision/classification/mobilenet
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_downloaded(MobileNet)?;

    // ndarray mini-batches of 3-channel RGB images of shape (N x 3 x H x W)
    let array = Array4::from_shape_fn((1, 3, 224, 224), |_| rand::random::<f32>());
    println!("{:?}", array);
    let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
        session.run([InputTensor::from_array(array.into_dyn())])?;

    let output: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
    let output_view = output.view().to_owned();

    // println!("{:?}", output_view);
    let (argmax, max_value) = argmax_1d(&output_view).unwrap();
    println!("{}, {}", argmax, max_value);

    Ok(())
}

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "caffe/layers/multi_class_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiClassAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  CHECK_EQ(bottom[0]->shape(label_axis_) % bottom[1]->shape(label_axis_), 0)
          <<  "number of predictions must be a multiple of labels!";

  num_classes_ = bottom[0]->shape(label_axis_) / bottom[1]->shape(label_axis_);
  num_classifiers_ = bottom[0]->shape(label_axis_) / num_classes_;
}

template <typename Dtype>
void MultiClassAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());

  // report accuracy per classifer
  top[0]->Reshape(num_classifiers_, 1, 1, 1);
}

template <typename Dtype>
void MultiClassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<Dtype> correct_predictions(num_classifiers_, Dtype(0));
  const Dtype* bottom_data = bottom[0]->cpu_data();

  // determine whether predictions of each classifier match
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int classifier_id = 0; classifier_id < num_classifiers_; ++classifier_id) {
      const int label_value = static_cast<int>(bottom[1]->data_at(n, classifier_id, 0, 0));
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_classes_);

      // find element that has the best predicted value and compare its id to label
      const Dtype * begin = bottom_data + bottom[0]->offset(n, classifier_id * num_classes_);
      const Dtype * end = bottom_data + bottom[0]->offset(n, (classifier_id + 1) * num_classes_);

      const Dtype * max_value = std::max_element(begin, end);
      int max_id = std::distance(begin, max_value);

      if (max_id == label_value) {
          correct_predictions[classifier_id] += 1;
      }
    }
  }

  // determine overall accuracy for each classifier
  for (int prediction_id = 0; prediction_id < correct_predictions.size(); ++prediction_id) {
    Dtype num_correct_predictions = correct_predictions[prediction_id];
    Dtype accuracy = num_correct_predictions / bottom[0]->num();
    top[0]->mutable_cpu_data()[prediction_id] = accuracy;
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MultiClassAccuracyLayer);
REGISTER_LAYER_CLASS(MultiClassAccuracy);

}  // namespace caffe

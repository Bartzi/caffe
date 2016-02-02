#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multi_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  CHECK_EQ(bottom[0]->shape(softmax_axis_) % bottom[1]->shape(softmax_axis_), 0)
          <<  "number of predictions must be a multiple of labels!";

  num_classes_ = bottom[0]->shape(softmax_axis_) / bottom[1]->shape(softmax_axis_);
  num_classifiers_ = bottom[0]->shape(softmax_axis_) / num_classes_;

  single_softmax_bottom.Reshape(
          bottom[0]->num() * num_classifiers_,
          num_classes_,
          1,
          1
  );

  // prepare softmax layer contained in this layer
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(&single_softmax_bottom);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  single_softmax_bottom.Reshape(
          bottom[0]->num() * num_classifiers_,
          num_classes_,
          1,
          1
  );

  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

  CHECK_EQ(bottom[0]->count() / num_classes_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "with integer values in {0, 1, ..., C-1}.";
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Dtype * single_softmax_bottom_data = single_softmax_bottom.mutable_cpu_data();
  vector<Dtype> loss_per_classifier(num_classifiers_, Dtype(0));

  Dtype loss = 0;
  // copy data of each classifier into softmax input blob
  caffe_copy(
      single_softmax_bottom.count(),
      bottom[0]->cpu_data(),
      single_softmax_bottom_data );

  // do softmax per classifier
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  for (int n = 0; n < single_softmax_bottom.num(); ++n) {
    // calculate loss and accumulate loss for each batch and each classifier
    const int label_value = static_cast<int>(bottom[1]->cpu_data()[n]);
    DCHECK_GE(label_value, 0);
    DCHECK_LT(label_value, prob_.shape(1));
    loss -= log(std::max(prob_.data_at(n, label_value, 0, 0), Dtype(FLT_MIN)));
  }

  // normalize and accumulate loss of each batch and classifier
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype * prob_data = prob_.mutable_cpu_data();

    for (int n = 0; n < prob_.num(); ++n) {
        // calculate gradient with respect to correct label
        const int label_value = static_cast<int>(bottom[1]->cpu_data()[n]);
        prob_data[prob_.offset(n, label_value, 0, 0)] -= 1;
    }

    // copy calculated diff to correct position in bottom
    caffe_copy(
        bottom[0]->count(),
        prob_data,
        bottom_diff
    );

    // scale with loss_weight
    Dtype loss_weight = top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASS(MultiSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(MultiSoftmaxWithLoss);

}  // namespace caffe

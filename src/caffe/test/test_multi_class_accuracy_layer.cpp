#include <cfloat>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/multi_class_accuracy_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class MultiClassAccuracyLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  MultiClassAccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_per_class_(new Blob<Dtype>()),
        top_k_(3) {
    blob_bottom_data_->Reshape(100, 10, 1, 1);
    blob_bottom_label_->Reshape(100, 1, 1, 1);
    FillBottoms();

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_per_class_vec_.push_back(blob_top_);
    blob_top_per_class_vec_.push_back(blob_top_per_class_);
  }

  virtual void FillBottoms() {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
          static_cast<caffe::rng_t*>(rng->generator());
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = (*prefetch_rng)() %
              (blob_bottom_data_->channels() / blob_bottom_label_->channels());
    }
  }

  virtual ~MultiClassAccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
    delete blob_top_per_class_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_per_class_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_per_class_vec_;
  int top_k_;
};

TYPED_TEST_CASE(MultiClassAccuracyLayerTest, TestDtypes);

TYPED_TEST(MultiClassAccuracyLayerTest, TestSetup) {
  LayerParameter layer_param;
  MultiClassAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(MultiClassAccuracyLayerTest, TestSetupMultipleClassifiers) {
  this->blob_bottom_label_->Reshape(100, 5, 1, 1);
  this->FillBottoms();
  LayerParameter layer_param;
  MultiClassAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_label_->channels());
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(MultiClassAccuracyLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  MultiClassAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++num_correct_labels;
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, 1e-4);
}

TYPED_TEST(MultiClassAccuracyLayerTest, TestForwardCPUMultipleClassifiers) {
  this->blob_bottom_label_->Reshape(100, 5, 1, 1);
  this->FillBottoms();

  LayerParameter layer_param;
  MultiClassAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  vector<TypeParam> correct_predictions(5, TypeParam(0));

  for (int i = 0; i < 100; ++i) {
    for (int classifier_id = 0; classifier_id < 5; ++classifier_id) {
      max_value = -FLT_MAX;
      max_id = 0;
      for (int j = 0; j < 2; ++j) {
        TypeParam value = this->blob_bottom_data_->data_at(
                    i,
                    classifier_id * 2 + j,
                    0,
                    0);
        if (value > max_value) {
          max_value = value;
          max_id = j;
        }
      }
      if (max_id == this->blob_bottom_label_->data_at(i, classifier_id, 0, 0)) {
        correct_predictions[classifier_id] += 1;
      }
    }
  }

  for (int prediction_id = 0;
       prediction_id < correct_predictions.size();
       ++prediction_id) {
      EXPECT_NEAR(this->blob_top_->data_at(prediction_id, 0, 0, 0),
          correct_predictions[prediction_id] / 100.0, 1e-4);
  }
}

}  // namespace caffe

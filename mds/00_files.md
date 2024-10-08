### 전체 코드 파일 구조
```
0_basics/240603_01_sampling.ipynb
0_basics/240603_02_hr_sampling_roc.ipynb
0_basics/240603_03_titanic_new_age.ipynb
0_basics/00_python_basics/00_string.ipynb
0_basics/00_python_basics/01_number.ipynb
0_basics/00_python_basics/02_variable.ipynb
0_basics/00_python_basics/03_string_function.ipynb
0_basics/00_python_basics/04_case.ipynb
0_basics/00_python_basics/05_list.ipynb
0_basics/00_python_basics/06_dict.ipynb
0_basics/00_python_basics/07_while.ipynb
0_basics/00_python_basics/07_while_baseball.ipynb
0_basics/00_python_basics/08_function.ipynb
0_basics/00_python_basics/08_function_full.ipynb
0_basics/00_python_basics/09_tuple_lambda_file.ipynb
0_basics/00_python_basics/10_exception.ipynb
0_basics/00_python_basics/11_class.ipynb
0_basics/00_python_basics/12_conda.ipynb
0_basics/00_python_basics/13_module.ipynb
0_basics/00_python_basics/14_API.ipynb
0_basics/00_python_basics/15_crawling.ipynb
0_basics/00_python_basics/16_numpy.ipynb
0_basics/00_python_basics/17_pandas.ipynb
0_basics/00_python_basics/18_plus_alpha.ipynb
0_basics/01_math/00_선형과 비선형.ipynb
0_basics/01_math/01_도함수 _계산.ipynb
0_basics/01_math/02_통계_1.ipynb
0_basics/01_math/03_통계_2.ipynb
0_basics/01_math/04_벡터화.ipynb
0_basics/01_math/05_벡터화 예시.ipynb
0_basics/01_math/06_embedding.ipynb
0_basics/01_math/07_공분산과_상관계수.ipynb
0_basics/02_web_crawling/00_이론.ipynb
0_basics/02_web_crawling/01_Beutifulsoup.ipynb
0_basics/02_web_crawling/02_BS_예제.ipynb
0_basics/02_web_crawling/03_XML 구조 실습.ipynb
1_machinelearing/00_이론/01_회귀_이론_.ipynb
1_machinelearing/00_이론/02_회귀_예시_iris 데이터.ipynb
1_machinelearing/00_이론/03_GridSearch.ipynb
1_machinelearing/00_이론/04_FeatureImportance.ipynb
1_machinelearing/00_이론/05_PCA_주성분분석.ipynb
1_machinelearing/01_Decision_Tree/00_dt_이론.ipynb
1_machinelearing/01_Decision_Tree/01_dt_분류.ipynb
1_machinelearing/01_Decision_Tree/02_dt_회귀.ipynb
1_machinelearing/01_Decision_Tree/03_dt_회귀예시.ipynb
1_machinelearing/02_RandomForest/00_RF_이론.ipynb
1_machinelearing/02_RandomForest/01_RF_iris.ipynb
1_machinelearing/02_RandomForest/02_RF_예시.ipynb
1_machinelearing/03_KNN/00_knn.ipynb
1_machinelearing/03_KNN/01_knn.ipynb
1_machinelearing/03_KNN/02_knn.ipynb
1_machinelearing/03_KNN/03_knn_예시.ipynb
1_machinelearing/04_SVM/_SVM_1 결정경계 그래프 예시.ipynb
1_machinelearing/04_SVM/_SVM_2 결정경계 그래프 예시.ipynb
1_machinelearing/04_SVM/_SVM_3 결정경계 그래프 예시.ipynb
1_machinelearing/05_Ensemble/00_Ensemble.ipynb
1_machinelearing/05_Ensemble/01_Ensemble_예시.ipynb
1_machinelearing/06_boosting/00_boosting.ipynb
1_machinelearing/06_boosting/01_XGBoost.ipynb
1_machinelearing/07_CrossValid/00_CrossValid.ipynb
1_machinelearing/07_CrossValid/01_CrossValid_2.ipynb
1_machinelearing/99_예시/00_cancer_.ipynb
1_machinelearing/99_예시/01_결측치,이상치.ipynb
1_machinelearing/99_예시/02_상관계수.ipynb
1_machinelearing/99_예시/03_iris_kaggle.ipynb
1_machinelearing/99_예시/04_iris_kaggle_2.ipynb
1_machinelearing/99_예시/05_iris_예시.ipynb
1_machinelearing/99_예시/06_iris_예시_2.ipynb
1_machinelearing/99_예시/07_hr_data.ipynb
1_machinelearing/99_예시/08_titanic_data.ipynb
1_machinelearing/99_예시/09_titanic_data_quiz.ipynb
1_machinelearing/99_예시/correlation, scaleing.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/00_다층 신경망.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/01_basic_2.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/02_basics.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/03_단층 신경망.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/04_house_price.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/05_IRIS_data.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/06_wine_data.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/07_cancer data.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/08_이진분류.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/09_digit.ipynb
2_DeepLearning/00_기본, 단층, 다층 신경망/10_단층, 다층_과제.ipynb
2_DeepLearning/01_딥러닝/00_활성화_함수.ipynb
2_DeepLearning/01_딥러닝/01_비용함수.ipynb
2_DeepLearning/01_딥러닝/02_비용함수_예제.ipynb
2_DeepLearning/01_딥러닝/03_역전파.ipynb
2_DeepLearning/01_딥러닝/04_역전파_예제.ipynb
2_DeepLearning/01_딥러닝/05_경사하강법.ipynb
2_DeepLearning/01_딥러닝/06_경사하강법_예제(Lr영향).ipynb
2_DeepLearning/01_딥러닝/07_옵티마이저.ipynb
2_DeepLearning/01_딥러닝/08_옵티마이저_예제.ipynb
2_DeepLearning/01_딥러닝/09_다양한 문제들.ipynb
2_DeepLearning/02_CNN/00_CNN.ipynb
2_DeepLearning/02_CNN/01_Cifar10.ipynb
2_DeepLearning/02_CNN/02_Mnist.ipynb
2_DeepLearning/02_CNN/04_Mnist_2.ipynb
2_DeepLearning/02_CNN/05_Cycle_Bicycle.ipynb
2_DeepLearning/03_RNN/00_순환신경망.ipynb
2_DeepLearning/03_RNN/01_.ipynb
2_DeepLearning/03_RNN/02_imdb_1.ipynb
2_DeepLearning/03_RNN/03_imdb_실습_return_seq_true_concat.ipynb
2_DeepLearning/03_RNN/04_imdb_실습_return_seq_true_mean.ipynb
2_DeepLearning/04_LSTM/00_LSTM.ipynb
2_DeepLearning/04_LSTM/01_LSTM2.ipynb
2_DeepLearning/04_LSTM/02_LSTM_실습.ipynb
2_DeepLearning/04_LSTM/03_LSTM_실습2.ipynb
2_DeepLearning/09_AutoEncoder/00_AE.ipynb
2_DeepLearning/09_AutoEncoder/01_Mnist.ipynb
2_DeepLearning/09_AutoEncoder/02_RatingData.ipynb
3_time_series/time_series_1_lag_feature.ipynb
3_time_series/00_시계열/00_시계열_1_이론.ipynb
3_time_series/00_시계열/01_광주데이터.ipynb
3_time_series/00_시계열/02_ARIMA_분석.ipynb
3_time_series/00_시계열/03_statsmodels_ARIMA.ipynb
3_time_series/00_시계열/04_기온_예측_.ipynb
3_time_series/01_시계열_snp500/00_snp500_1.ipynb
3_time_series/01_시계열_snp500/01_snp500_2.ipynb
3_time_series/02_시퀀스/00_크롤링.ipynb
3_time_series/02_시퀀스/01_모델링(yes24).ipynb
3_time_series/02_시퀀스/02_EDA.ipynb
3_time_series/02_시퀀스/03_ 모델링 예시.ipynb
3_time_series/02_시퀀스/04_클러스터링1.ipynb
3_time_series/02_시퀀스/05_클러스터링2.ipynb
3_time_series/02_시퀀스/06_클러스터링3_직접.ipynb
3_time_series/timeSeries/20240525_trend.ipynb
3_time_series/timeSeries/20240526_seasonal.ipynb
3_time_series/timeSeries/20240526_seasonal_exp.ipynb
3_time_series/timeSeries/240602.ipynb
4_visualization/00_개요.ipynb
4_visualization/240520_folium.ipynb
4_visualization/240521_02_crime.ipynb
4_visualization/240522_01_hr.ipynb
4_visualization/240522_02_quatile.ipynb
4_visualization/240523_02__Matplotlib_.ipynb
4_visualization/240523_03_titanic.ipynb
4_visualization/240524_00_boxplot.ipynb
4_visualization/240524_01_correlation.ipynb
4_visualization/240524_02_scaling.ipynb
4_visualization/240524_03_PCA.ipynb
4_visualization/240524_04_hr.ipynb
4_visualization/240524_05_hr.ipynb
4_visualization/240527_00_hr.ipynb
4_visualization/_PCA_아이리스 데이터.ipynb
4_visualization/시각화_10_folium_개인 연습_1.ipynb
4_visualization/시각화_10_folium_개인 연습_2.ipynb
4_visualization/시각화_2_혼란한 Matplotlib에서 질서 찾기.ipynb
4_visualization/시각화_3_시각화 해보기.ipynb
4_visualization/시각화_4_ 그래프만 모아서.ipynb
4_visualization/시각화_5_times_series.ipynb
4_visualization/시각화_6_box_plot_IQR.ipynb
4_visualization/시각화_7_cv2_그림.ipynb
4_visualization/시각화_8_folium_1.ipynb
4_visualization/시각화_9_folium_2_과제.ipynb
5_layers/00_SoftMax.ipynb
5_layers/01_batch_norm.ipynb
5_layers/02_early_stop.ipynb
5_layers/03_ReceptiveField.ipynb
5_layers/04_dropout.ipynb
6_metrics/240531_01_confusion_matrix.ipynb
6_metrics/240531_02_confusion_matrix.ipynb
6_metrics/_Confusion Matrix.ipynb
6_metrics/머신러닝_12_Confusion_mat.ipynb
7_pretrained_model/01_크롤링_후_CNN_분류/_00_네이버_크롤링.ipynb
7_pretrained_model/01_크롤링_후_CNN_분류/_01_산_바다_분류.ipynb
7_pretrained_model/01_크롤링_후_CNN_분류/_02_cifar10_분류.ipynb
7_pretrained_model/01_크롤링_후_CNN_분류/_03_FNN.MNIST_분류.ipynb
7_pretrained_model/01_크롤링_후_CNN_분류/_04_image_crawling.ipynb
7_pretrained_model/01_크롤링_후_CNN_분류/_05_크롤링_후_다중분류.ipynb
7_pretrained_model/02_데이터_증강_및_모델_저장/_01_AE_데이터_증강.ipynb
7_pretrained_model/02_데이터_증강_및_모델_저장/_02_AE_데이터_증강_폴더_단위.ipynb
7_pretrained_model/02_데이터_증강_및_모델_저장/_03_AE_데이터_증강_재훈련.ipynb
7_pretrained_model/02_데이터_증강_및_모델_저장/_04_keras_이미지_변형_생성.ipynb
7_pretrained_model/02_데이터_증강_및_모델_저장/_05_keras_이미지_변형_생성_폴더_단위.ipynb
7_pretrained_model/02_데이터_증강_및_모델_저장/_06_AE증강 vs keras증강 훈련 결과.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/1_abstract/_01_LeNet_MNIST_Call.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/1_abstract/_02_LeNet_MNIST_변형.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/1_abstract/_03_ResNet_로드_예측.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/2_AlexNet/_01_AlexNet_구현하기.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/2_AlexNet/_02_AlexNet_정답코드.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/3_VGG_inception/_01_VGG16_FeatureMap.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/3_VGG_inception/_02_VGG16.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/3_VGG_inception/_04_VGG16_로드_예측.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/4_inception(ggl)Net/_01_InceptionNet_Architecture.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/4_inception(ggl)Net/_02_PreTrained_InceptionNet.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/5_ResNet/_01_PreTrained_ResNet_Design.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/5_ResNet/_02_Pre_trained_ResNet50.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/6_MobileNet/_02_MobileNet_depth_wise.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/6_MobileNet/_03_MobileNet.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/7_DenseNet/7.03.PreTrained_DenseNet.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/8_EfficientNet/7.03.PreTrained_EfficientNet.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/9_img_prep/7.04.Pretrained_Test_Image_preprocessing_p.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/9_img_prep/7.04.Pre_trained_Image_preprocessing.ipynb
7_pretrained_model/03_CNN_base_pretrained_model/9_img_prep/7.05.Pretrained_Test_Image_preprocessing.ipynb
7_pretrained_model/04_CNN_피쳐맵_및_전이학습/01_Inception_to_ML.ipynb
7_pretrained_model/04_CNN_피쳐맵_및_전이학습/02_FetureMap.ipynb
7_pretrained_model/04_CNN_피쳐맵_및_전이학습/03_MobileNet_전이 학습.ipynb
7_pretrained_model/04_CNN_피쳐맵_및_전이학습/04_MobileNet_일부_파인튠.ipynb
7_pretrained_model/05_RNN/5.0.SeqData_Anal.ipynb
7_pretrained_model/05_RNN/5.0.SeqData_Anal_AutoArima.ipynb
7_pretrained_model/06_Seq2Seq/_02_LM_seq2seq.ipynb
7_pretrained_model/07_Transformer/_00_positioning_embedding.ipynb
7_pretrained_model/07_Transformer/_01_self_Attention.ipynb
7_pretrained_model/07_Transformer/_02_Transformer_imdb.ipynb
7_pretrained_model/08_Bert/7.06.LM_BERTopic_En.ipynb
7_pretrained_model/09_Object_Detection/_01_OD_NMS.ipynb
7_pretrained_model/09_Object_Detection/_02_OD_SSD_mAP.ipynb
7_pretrained_model/09_Object_Detection/_03_OD_SSD_example.ipynb
7_pretrained_model/09_Object_Detection/_04_ReceptiveField.ipynb
7_pretrained_model/09_Object_Detection/_05_Semantic_Upsampling_Transposed.ipynb
7_pretrained_model/09_Object_Detection/_06_Semantic_Segmentation.ipynb
7_pretrained_model/09_Object_Detection/_07_Semantic_Performance.ipynb
7_pretrained_model/09_Object_Detection/_08_OD_RCNN_Offset.ipynb
7_pretrained_model/10_YOLO/7.04.self.ipynb
7_pretrained_model/10_YOLO/7.05.OD_yolo_for_class.ipynb
7_pretrained_model/10_YOLO/7.05.z_주석.ipynb
7_pretrained_model/11_ChatGPT_API/8.01.LLM_ChatGPT_API.ipynb
7_pretrained_model/11_ChatGPT_API/8.01.LLM_ChatGPT_API_Chatbot.ipynb
7_pretrained_model/11_모델 응용/8.01.Model_Util_CNN_TextAna.ipynb
7_pretrained_model/12_Private_chat/8.02.Llmal.ipynb
7_pretrained_model/12_Private_chat/8.02.LnagChain_Text_Summarize.ipynb
7_pretrained_model/12_Private_chat/_01_GPT_2.ipynb
7_pretrained_model/12_Private_chat/_02_llama_3_bllossom.ipynb
7_pretrained_model/13_NER/8.02.NER.ipynb
7_pretrained_model/14_VQA/7.02.VQA.ipynb
8.ETC/7.02.VQA.ipynb
8.ETC/_간이세미나 1. 딥러닝에서 배치 크기의 역할.ipynb
8.ETC/_간이세미나 10 인공지능의 윤리적 고려사항.ipynb
8.ETC/_간이세미나 2. 텐서 자료형.ipynb
8.ETC/_간이세미나 3. 옵티마이저 비교.ipynb
8.ETC/_간이세미나 4 인공지능의 편향성과 차별.ipynb
8.ETC/_간이세미나 5. 인공지능의 창의성과 저작권.ipynb
8.ETC/_간이세미나 6 딥러닝 모델의 해석가능성.ipynb
8.ETC/_간이세미나 7 퍼셉트론.ipynb
8.ETC/_간이세미나 8 데이터 활용과 개인정보 보호.ipynb
8.ETC/_간이세미나 9 활성화 함수.ipynb
8.ETC/101_발표/발표_1_transformer.ipynb
8.ETC/101_발표/발표_2_alex_net.ipynb
8.ETC/99_과제 및 퀴즈/0724/0724_weekly_quiz_박진형.ipynb
8.ETC/99_과제 및 퀴즈/0724 위클리 퀴즈 의료/0724_weekly_quiz_박진형.ipynb
8.ETC/99_과제 및 퀴즈/0729_CNN_pretrained/0. 계획.ipynb
8.ETC/99_과제 및 퀴즈/0729_CNN_pretrained/1_1_transfer_with_ML_1_codes.ipynb
8.ETC/99_과제 및 퀴즈/0729_CNN_pretrained/2_1_transfer_mobilenet.ipynb
8.ETC/99_과제 및 퀴즈/0729_CNN_pretrained/3_1_finetune_mobilenet.ipynb
8.ETC/99_과제 및 퀴즈/0729_CNN_pretrained/4_1_from_scratch_NN.ipynb
8.ETC/99_과제 및 퀴즈/0729_CNN_pretrained/5_1_just_use_mobile.ipynb
8.ETC/99_과제 및 퀴즈/0729_CNN_pretrained/6_1_모든모델 불러와 비교하기.ipynb
8.ETC/99_과제 및 퀴즈/072x 과제/0. 계획.ipynb
8.ETC/99_과제 및 퀴즈/072x 과제/1_1_transfer_with_ML_1_codes.ipynb
8.ETC/99_과제 및 퀴즈/072x 과제/2_1_transfer_mobilenet.ipynb
8.ETC/99_과제 및 퀴즈/072x 과제/3_1_finetune_mobilenet.ipynb
8.ETC/99_과제 및 퀴즈/072x 과제/4_1_from_scratch_NN.ipynb
8.ETC/99_과제 및 퀴즈/0809 위클리 퀴즈 YOLO/week11_박진형.ipynb
8.ETC/99_과제 및 퀴즈/99_과제/week10_박진형.ipynb
8.ETC/99_과제 및 퀴즈/99_과제/week1_박진형.ipynb
8.ETC/99_과제 및 퀴즈/99_과제/week2_박진형.ipynb
8.ETC/99_과제 및 퀴즈/99_과제/week3_박진형.ipynb
8.ETC/99_과제 및 퀴즈/99_과제/week4_박진형.ipynb
8.ETC/99_과제 및 퀴즈/99_과제/week8_박진형.ipynb
8.ETC/99_과제 및 퀴즈/99_과제/week9_박진형.ipynb
8.ETC/99_과제 및 퀴즈/과제_박진형/0. 계획.ipynb
8.ETC/99_과제 및 퀴즈/과제_박진형/1_1_transfer_with_ML_1_codes.ipynb
8.ETC/99_과제 및 퀴즈/과제_박진형/2_1_transfer_mobilenet.ipynb
8.ETC/99_과제 및 퀴즈/과제_박진형/3_1_finetune_mobilenet.ipynb
8.ETC/99_과제 및 퀴즈/과제_박진형/4_1_from_scratch_NN.ipynb
8.ETC/99_과제 및 퀴즈/과제_박진형/5_1_just_use_mobile.ipynb
8.ETC/99_과제 및 퀴즈/과제_박진형/6_1_모든모델 불러와 비교하기.ipynb
8.ETC/kdt1/240604_00_mnist.ipynb
8.ETC/kdt1/240604_01_optimizer.ipynb
```
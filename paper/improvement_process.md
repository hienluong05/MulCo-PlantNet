# Seesaw Loss

```
Accuracy:          0.7840 (95% CI: 0.7360 - 0.8320)
Macro F1-Score:    0.7611 (95% CI: 0.6951 - 0.8050)
Balanced Accuracy: 0.7942 (95% CI: 0.7445 - 0.8412)

-------------------------------------------------------
 CLASSIFICATION REPORT CHI TIẾT TỪNG LỚP BỆNH
                                      precision    recall  f1-score   support

                     Apple_Scab_Leaf     0.7500    1.0000    0.8571         9
                          Apple_leaf     0.7273    0.8000    0.7619        10
                     Apple_rust_leaf     1.0000    0.7000    0.8235        10
                    Bell_pepper_leaf     0.6667    1.0000    0.8000         8
               Bell_pepper_leaf_spot     0.7273    0.8889    0.8000         9
                      Blueberry_leaf     0.9091    0.9091    0.9091        11
                         Cherry_leaf     1.0000    0.9000    0.9474        10
                 Corn_Gray_leaf_spot     0.3750    0.7500    0.5000         4
                    Corn_leaf_blight     0.8571    0.5000    0.6316        12
                      Corn_rust_leaf     0.9091    1.0000    0.9524        10
                          Peach_leaf     0.9231    1.0000    0.9600        12
            Potato_leaf_early_blight     1.0000    0.8750    0.9333         8
             Potato_leaf_late_blight     1.0000    1.0000    1.0000         9
                      Raspberry_leaf     0.7143    0.3571    0.4762        14
                       Soyabean_leaf     0.3000    0.3750    0.3333         8
          Squash_Powdery_mildew_leaf     0.8571    0.8571    0.8571         7
                     Strawberry_leaf     0.8333    0.7143    0.7692         7
            Tomato_Early_blight_leaf     1.0000    1.0000    1.0000         6
           Tomato_Septoria_leaf_spot     0.8889    1.0000    0.9412         8
                         Tomato_leaf     0.7000    0.7778    0.7368         9
          Tomato_leaf_bacterial_spot     0.7500    0.7500    0.7500         8
             Tomato_leaf_late_blight     0.5000    0.2500    0.3333         8
            Tomato_leaf_mosaic_virus     0.7273    0.8000    0.7619        10
            Tomato_leaf_yellow_virus     1.0000    0.4000    0.5714        10
                    Tomato_mold_leaf     0.9375    1.0000    0.9677        15
Tomato_two_spotted_spider_mites_leaf     0.5714    0.8000    0.6667         5
                          grape_leaf     0.7143    0.8333    0.7692        12
                grape_leaf_black_rot     0.3333    1.0000    0.5000         1

                            accuracy                         0.7840       250
                           macro avg     0.7740    0.7942    0.7611       250
                        weighted avg     0.8064    0.7840    0.7768       250
```

# CB Focal Loss - beta = 0.999

```
Accuracy:          0.7720 (95% CI: 0.7200 - 0.8240)
Macro F1-Score:    0.7651 (95% CI: 0.7060 - 0.8028)
Balanced Accuracy: 0.7902 (95% CI: 0.7428 - 0.8306)

-------------------------------------------------------
 CLASSIFICATION REPORT CHI TIẾT TỪNG LỚP BỆNH
                                      precision    recall  f1-score   support

                     Apple_Scab_Leaf     0.9000    1.0000    0.9474         9
                          Apple_leaf     0.8182    0.9000    0.8571        10
                     Apple_rust_leaf     1.0000    0.7000    0.8235        10
                    Bell_pepper_leaf     0.7273    1.0000    0.8421         8
               Bell_pepper_leaf_spot     0.8182    1.0000    0.9000         9
                      Blueberry_leaf     1.0000    0.9091    0.9524        11
                         Cherry_leaf     0.9091    1.0000    0.9524        10
                 Corn_Gray_leaf_spot     0.3077    1.0000    0.4706         4
                    Corn_leaf_blight     1.0000    0.4167    0.5882        12
                      Corn_rust_leaf     1.0000    0.8000    0.8889        10
                          Peach_leaf     0.9231    1.0000    0.9600        12
            Potato_leaf_early_blight     0.7778    0.8750    0.8235         8
             Potato_leaf_late_blight     1.0000    0.8889    0.9412         9
                      Raspberry_leaf     0.6250    0.3571    0.4545        14
                       Soyabean_leaf     0.3333    0.3750    0.3529         8
          Squash_Powdery_mildew_leaf     0.8750    1.0000    0.9333         7
                     Strawberry_leaf     1.0000    0.7143    0.8333         7
            Tomato_Early_blight_leaf     1.0000    1.0000    1.0000         6
           Tomato_Septoria_leaf_spot     1.0000    1.0000    1.0000         8
                         Tomato_leaf     0.5714    0.8889    0.6957         9
          Tomato_leaf_bacterial_spot     0.6667    0.5000    0.5714         8
             Tomato_leaf_late_blight     0.0000    0.0000    0.0000         8
            Tomato_leaf_mosaic_virus     0.6667    0.8000    0.7273        10
            Tomato_leaf_yellow_virus     0.8750    0.7000    0.7778        10
                    Tomato_mold_leaf     0.9375    1.0000    0.9677        15
Tomato_two_spotted_spider_mites_leaf     0.5000    0.8000    0.6154         5
                          grape_leaf     0.6000    0.5000    0.5455        12
                grape_leaf_black_rot     1.0000    1.0000    1.0000         1

                            accuracy                         0.7720       250
                           macro avg     0.7797    0.7902    0.7651       250
                        weighted avg     0.7931    0.7720    0.7646       250
```

# CB Focal Loss - beta = 0.99

```
Accuracy:          0.8160 (95% CI: 0.7680 - 0.8680)
Macro F1-Score:    0.8091 (95% CI: 0.7537 - 0.8441)
Balanced Accuracy: 0.8238 (95% CI: 0.7760 - 0.8674)

-------------------------------------------------------
 CLASSIFICATION REPORT CHI TIẾT TỪNG LỚP BỆNH
                                      precision    recall  f1-score   support

                     Apple_Scab_Leaf     0.9000    1.0000    0.9474         9
                          Apple_leaf     0.8182    0.9000    0.8571        10
                     Apple_rust_leaf     1.0000    0.7000    0.8235        10
                    Bell_pepper_leaf     0.7778    0.8750    0.8235         8
               Bell_pepper_leaf_spot     0.7500    1.0000    0.8571         9
                      Blueberry_leaf     1.0000    0.9091    0.9524        11
                         Cherry_leaf     1.0000    1.0000    1.0000        10
                 Corn_Gray_leaf_spot     0.2727    0.7500    0.4000         4 #
                    Corn_leaf_blight     0.8333    0.4167    0.5556        12 #
                      Corn_rust_leaf     1.0000    0.9000    0.9474        10
                          Peach_leaf     1.0000    1.0000    1.0000        12
            Potato_leaf_early_blight     0.8000    1.0000    0.8889         8
             Potato_leaf_late_blight     1.0000    1.0000    1.0000         9
                      Raspberry_leaf     0.7500    0.6429    0.6923        14
                       Soyabean_leaf     0.2857    0.2500    0.2667         8 #
          Squash_Powdery_mildew_leaf     0.8750    1.0000    0.9333         7
                     Strawberry_leaf     1.0000    0.8571    0.9231         7
            Tomato_Early_blight_leaf     1.0000    1.0000    1.0000         6
           Tomato_Septoria_leaf_spot     1.0000    1.0000    1.0000         8
                         Tomato_leaf     0.7500    0.6667    0.7059         9
          Tomato_leaf_bacterial_spot     0.7500    0.7500    0.7500         8
             Tomato_leaf_late_blight     0.5000    0.2500    0.3333         8 #
            Tomato_leaf_mosaic_virus     0.7273    0.8000    0.7619        10
            Tomato_leaf_yellow_virus     0.9000    0.9000    0.9000        10
                    Tomato_mold_leaf     1.0000    0.8667    0.9286        15
Tomato_two_spotted_spider_mites_leaf     0.5714    0.8000    0.6667         5
                          grape_leaf     0.6667    0.8333    0.7407        12
                grape_leaf_black_rot     1.0000    1.0000    1.0000         1

                            accuracy                         0.8160       250
                           macro avg     0.8189    0.8238    0.8091       250
                        weighted avg     0.8341    0.8160    0.8143       250
```

# CB Focal Loss - beta = 0.99 + MLP + 2 fusion blocks

```
Accuracy:          0.8560 (95% CI: 0.8119 - 0.8960)
Macro F1-Score:    0.8487 (95% CI: 0.7925 - 0.8823)
Balanced Accuracy: 0.8586 (95% CI: 0.8145 - 0.8960)

-------------------------------------------------------
 CLASSIFICATION REPORT CHI TIẾT TỪNG LỚP BỆNH
                                      precision    recall  f1-score   support

                     Apple_Scab_Leaf     0.7500    1.0000    0.8571         9
                          Apple_leaf     1.0000    0.9000    0.9474        10
                     Apple_rust_leaf     1.0000    0.9000    0.9474        10
                    Bell_pepper_leaf     0.8889    1.0000    0.9412         8
               Bell_pepper_leaf_spot     0.7500    1.0000    0.8571         9
                      Blueberry_leaf     1.0000    0.9091    0.9524        11
                         Cherry_leaf     0.8750    0.7000    0.7778        10
                 Corn_Gray_leaf_spot     0.4286    0.7500    0.5455         4 #
                    Corn_leaf_blight     0.7273    0.6667    0.6957        12 #
                      Corn_rust_leaf     1.0000    0.8000    0.8889        10
                          Peach_leaf     0.9231    1.0000    0.9600        12
            Potato_leaf_early_blight     1.0000    0.8750    0.9333         8
             Potato_leaf_late_blight     0.8889    0.8889    0.8889         9
                      Raspberry_leaf     0.8667    0.9286    0.8966        14
                       Soyabean_leaf     0.7500    0.7500    0.7500         8
          Squash_Powdery_mildew_leaf     1.0000    1.0000    1.0000         7
                     Strawberry_leaf     0.8571    0.8571    0.8571         7
            Tomato_Early_blight_leaf     1.0000    1.0000    1.0000         6
           Tomato_Septoria_leaf_spot     1.0000    1.0000    1.0000         8
                         Tomato_leaf     0.7500    1.0000    0.8571         9
          Tomato_leaf_bacterial_spot     0.8000    0.5000    0.6154         8 #
             Tomato_leaf_late_blight     0.5000    0.2500    0.3333         8 #
            Tomato_leaf_mosaic_virus     0.8333    1.0000    0.9091        10
            Tomato_leaf_yellow_virus     0.8750    0.7000    0.7778        10
                    Tomato_mold_leaf     0.8824    1.0000    0.9375        15
Tomato_two_spotted_spider_mites_leaf     0.8333    1.0000    0.9091         5
                          grape_leaf     0.8000    0.6667    0.7273        12
                grape_leaf_black_rot     1.0000    1.0000    1.0000         1

                            accuracy                         0.8560       250
                           macro avg     0.8564    0.8586    0.8487       250
                        weighted avg     0.8610    0.8560    0.8505       250
```

# Như trên nhưng thay GAP bằng GeM

```
Accuracy:          0.8720 (95% CI: 0.8320 - 0.9080)
Macro F1-Score:    0.8625 (95% CI: 0.8080 - 0.8949)
Balanced Accuracy: 0.8701 (95% CI: 0.8306 - 0.9067)

-------------------------------------------------------
 CLASSIFICATION REPORT CHI TIẾT TỪNG LỚP BỆNH
                                      precision    recall  f1-score   support

                     Apple_Scab_Leaf     0.8182    1.0000    0.9000         9
                          Apple_leaf     1.0000    0.8000    0.8889        10
                     Apple_rust_leaf     0.9091    1.0000    0.9524        10
                    Bell_pepper_leaf     0.8889    1.0000    0.9412         8
               Bell_pepper_leaf_spot     0.8182    1.0000    0.9000         9
                      Blueberry_leaf     1.0000    0.9091    0.9524        11
                         Cherry_leaf     1.0000    0.9000    0.9474        10
                 Corn_Gray_leaf_spot     0.3750    0.7500    0.5000         4
                    Corn_leaf_blight     0.7000    0.5833    0.6364        12
                      Corn_rust_leaf     1.0000    0.8000    0.8889        10
                          Peach_leaf     0.9231    1.0000    0.9600        12
            Potato_leaf_early_blight     1.0000    0.8750    0.9333         8
             Potato_leaf_late_blight     1.0000    1.0000    1.0000         9
                      Raspberry_leaf     0.8571    0.8571    0.8571        14
                       Soyabean_leaf     0.7143    0.6250    0.6667         8
          Squash_Powdery_mildew_leaf     0.8750    1.0000    0.9333         7
                     Strawberry_leaf     1.0000    0.7143    0.8333         7
            Tomato_Early_blight_leaf     1.0000    1.0000    1.0000         6
           Tomato_Septoria_leaf_spot     1.0000    1.0000    1.0000         8
                         Tomato_leaf     0.8182    1.0000    0.9000         9
          Tomato_leaf_bacterial_spot     0.8000    0.5000    0.6154         8
             Tomato_leaf_late_blight     1.0000    0.2500    0.4000         8
            Tomato_leaf_mosaic_virus     0.7692    1.0000    0.8696        10
            Tomato_leaf_yellow_virus     0.8889    0.8000    0.8421        10
                    Tomato_mold_leaf     0.8333    1.0000    0.9091        15
Tomato_two_spotted_spider_mites_leaf     1.0000    1.0000    1.0000         5
                          grape_leaf     0.8571    1.0000    0.9231        12
                grape_leaf_black_rot     1.0000    1.0000    1.0000         1

                            accuracy                         0.8720       250
                           macro avg     0.8873    0.8701    0.8625       250
                        weighted avg     0.8875    0.8720    0.8654       250
```

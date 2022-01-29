# Usage

* Run `feature_engineering_sliding_window_2.py` to generate testing data
* Run `test.py`

In the output/only-test folder, you will get two csv files.
* baseline_submission_1998.csv: model output
* baseline_submission_post_preprocess_1998.csv: postprocessed output

baseline_submission_post_preprocess_1998.csv will get 86.3012 and ranked 3rd in the B Leaderboard.


# Training


* Extract wsdm_data.tar.gz to data Foler
* move test-a.csv and test_B.csv into data Foler
* Run `feature_engineering_sliding_window_1.py` to generate training data
* Run `feature_engineering_sliding_window_2.py` to generate testing data
* Run `Model_Sliding_Window.py`

Training will cost a lot of time. We used A100 and trained almost 5 hours.
After training, the output result will be saved in the subfolder of `./checkpoints/paddle/`


# Solution
TO_DO

# Notes🎂
* 为什么要跑两次数据集生成代码 
> 这是因为实验发现，即使在B阶段，使用A阶段的测试集来生成的数据集效果依然比使用test-B.csv生成的数据集效果好一点（使用test-B.csv生成的数据集在相同参数下训练得到的分数也足以获得第三名）


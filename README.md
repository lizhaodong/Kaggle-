Kaggle竞赛方案盘点

到目前为止，Kaggle 平台上面已经举办了大大小小不同的赛事，覆盖图像分类，销量预估，搜索相关性，点击率预估等应用场景。在不少的比赛中，获胜者都会把自己的方案开源出来，并且非常乐于分享比赛经验和技巧心得。这些开源方案和经验分享对于广大的新手和老手来说，是入门和进阶非常好的参考资料。以下笔者结合自身的背景和兴趣，对不同场景的竞赛开源方案作一个简单的盘点，总结其常用的方法和工具，以期启发思路。



3.1 图像分类

3.1.1 任务名称

National Data Science Bowl

3.1.2 任务详情

随着深度学习在视觉图像领域获得巨大成功，Kaggle 上面出现了越来越多跟视觉图像相关的比赛。这些比赛的发布吸引了众多参赛选手，探索基于深度学习的方法来解决垂直领域的图像问题。NDSB就是其中一个比较早期的图像分类相关的比赛。这个比赛的目标是利用提供的大量的海洋浮游生物的二值图像，通过构建模型，从而实现自动分类。

3.1.3 获奖方案

● 1st place：

Cyclic Pooling + Rolling Feature Maps + Unsupervised and Semi-Supervised Approaches。值得一提的是，这个队伍的主力队员也是Galaxy Zoo行星图像分类比赛的第一名，其也是Theano中基于FFT的Fast Conv的开发者。在两次比赛中，使用的都是 Theano，而且用的非常溜。方案链接：

http://benanne.github.io/2015/03/17/plankton.html

● 2nd place：

Deep CNN designing theory + VGG-like model + RReLU。这个队伍阵容也相当强大，有前MSRA 的研究员Xudong Cao，还有大神Tianqi Chen，Naiyan Wang，Bing XU等。Tianqi 等大神当时使用的是 CXXNet（MXNet 的前身），也在这个比赛中进行了推广。Tianqi 大神另外一个大名鼎鼎的作品就是 XGBoost，现在 Kaggle 上面几乎每场比赛的 Top 10 队伍都会使用。方案链接：

https://www.kaggle.com/c/datasciencebowl/discussion/13166

● 17th place：

Realtime data augmentation + BN + PReLU。方案链接：

https://github.com/ChenglongChen/caffe-windows

3.1.4 常用工具

▲ Theano: http://deeplearning.net/software/theano/

▲ Keras: https://keras.io/

▲ Cuda-convnet2: 

https://github.com/akrizhevsky/cuda-convnet2

▲ Caffe: http://caffe.berkeleyvision.org/

▲ CXXNET: https://github.com/dmlc/cxxnet

▲ MXNet: https://github.com/dmlc/mxnet

▲ PaddlePaddle: http://www.paddlepaddle.org/cn/index.html



3.2 销量预估

3.2.1 任务名称

Walmart Recruiting - Store Sales Forecasting

3.2.2 任务详情

Walmart 提供 2010-02-05 到 2012-11-01 期间的周销售记录作为训练数据，需要参赛选手建立模型预测 2012-11-02 到 2013-07-26 周销售量。比赛提供的特征数据包含：Store ID, Department ID, CPI，气温，汽油价格，失业率，是否节假日等。

3.2.3 获奖方案

● 1st place：

Time series forecasting method: stlf + arima + ets。主要是基于时序序列的统计方法，大量使用了 Rob J Hyndman 的 forecast R 包。方案链接：

https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8125

● 2nd place：

Time series forecasting + ML: arima + RF + LR + PCR。时序序列的统计方法+传统机器学习方法的混合；方案链接：

https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023

● 16th place：

Feature engineering + GBM。方案链接：

https://github.com/ChenglongChen/Kaggle_Walmart-Recruiting-Store-Sales-Forecasting

3.2.4 常用工具

▲ R forecast package: 

https://cran.r-project.org/web/packages/forecast/index.html

▲ R GBM package:

https://cran.r-project.org/web/packages/gbm/index.html



3.3 搜索相关性

3.3.1 任务名称

CrowdFlower Search Results Relevance

3.3.2 任务详情

比赛要求选手利用约几万个 (query, title, description) 元组的数据作为训练样本，构建模型预测其相关性打分 {1, 2, 3, 4}。比赛提供了 query, title和description的原始文本数据。比赛使用 Quadratic Weighted Kappa 作为评估标准，使得该任务有别于常见的回归和分类任务。

3.3.3 获奖方案

● 1st place：

Data Cleaning + Feature Engineering + Base Model + Ensemble。对原始文本数据进行清洗后，提取了属性特征，距离特征和基于分组的统计特征等大量的特征，使用了不同的目标函数训练不同的模型（回归，分类，排序等），最后使用模型集成的方法对不同模型的预测结果进行融合。方案链接：

https://github.com/ChenglongChen/Kaggle_CrowdFlower

● 2nd place：A Similar Workflow

● 3rd place： A Similar Workflow

3.3.4 常用工具

▲ NLTK: http://www.nltk.org/

▲ Gensim: https://radimrehurek.com/gensim/

▲ XGBoost: https://github.com/dmlc/xgboost

▲ RGF: https://github.com/baidu/fast_rgf



3.4 点击率预估 I

3.4.1 任务名称

Criteo Display Advertising Challenge

3.4.2 任务详情

经典的点击率预估比赛。该比赛中提供了7天的训练数据，1 天的测试数据。其中有13 个整数特征，26 个类别特征，均脱敏，因此无法知道具体特征含义。

3.4.3 获奖方案

● 1st place：

GBDT 特征编码 + FFM。台大的队伍，借鉴了Facebook的方案 [6]，使用 GBDT 对特征进行编码，然后将编码后的特征以及其他特征输入到 Field-aware Factorization Machine（FFM） 中进行建模。方案链接：

https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10555

● 3rd place：

Quadratic Feature Generation + FTRL。传统特征工程和 FTRL 线性模型的结合。方案链接：

https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10534

● 4th place：

Feature Engineering + Sparse DNN

3.4.4 常用工具

▲ Vowpal Wabbit: https://github.com/JohnLangford/vowpal_wabbit

▲ XGBoost: https://github.com/dmlc/xgboost

▲ LIBFFM: http://www.csie.ntu.edu.tw/~r01922136/libffm/



3.5 点击率预估 II

3.5.1 任务名称

Avazu Click-Through Rate Prediction

3.5.2 任务详情

点击率预估比赛。提供了 10 天的训练数据，1 天的测试数据，并且提供时间，banner 位置，site, app, device 特征等，8个脱敏类别特征。

3.5.3 获奖方案

● 1st place：Feature Engineering + FFM + Ensemble。还是台大的队伍，这次比赛，他们大量使用了 FFM，并只基于 FFM 进行集成。方案链接：

https://www.kaggle.com/c/avazu-ctr-prediction/discussion/12608

● 2nd place：Feature Engineering + GBDT 特征编码 + FFM + Blending。Owenzhang（曾经长时间雄霸 Kaggle 排行榜第一）的竞赛方案。Owenzhang 的特征工程做得非常有参考价值。方案链接：

https://github.com/owenzhang/kaggle-avazu

3.5.4 常用工具

▲ LIBFFM: http://www.csie.ntu.edu.tw/~r01922136/libffm/

▲ XGBoost: https://github.com/dmlc/xgboost



 4.

参考资料

[1] Owenzhang 的分享： Tips for Data Science Competitions

[2] Algorithms for Hyper-Parameter Optimization

[3] MLWave博客：Kaggle Ensembling Guide

[4] Jeong-Yoon Lee 的分享：Winning Data Science Competitions

[5] Ensemble Selection from Libraries of Models

[6] Practical Lessons from Predicting Clicks on Ads at Facebook



5.

结语

作为曾经的学生党，十分感激和庆幸有 Kaggle 这样的平台，提供了不同领域极具挑战的任务以及丰富多样的数据。让我这种空有满（yi）腔（xie）理（wai）论（li）的数据挖掘小白，可以在真实的问题场景和业务数据中进行实操练手，提升自己的数据挖掘技能，一不小心，还能拿名次，赢奖金。如果你也跃跃欲试，不妨选一个合适的任务，开启数据挖掘之旅吧。附腾讯广告算法大赛传送门：http://algo.tpai.qq.com


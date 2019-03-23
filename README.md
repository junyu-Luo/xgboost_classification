# 用xgboost进行预测（分类）


* 项目需要采用过 one class SVN 和 lasso，效果不佳，可以忽略这两个
* 将训练数据处理成与 `./data/` 相同的规范格式
* 执行 `python xgb.py` 命令可得到model文件
* 执行 `python find_best_params.py` 命令寻找最佳参数
* 执行 `python correlation_analysis.py` 命令分析重要因素
* `python predict_api.py` 命令是封装好的调用接口

main
----tation_get
--------filter_trial
描述：
    main: 训练和测试主函数
    tation_get: 获取对应的T1,T2状态
    filter_trail: 带通滤波


edf_extract_all: 解析edf文件，生成npy格式数据
描述：edf_extract_all解析S001中的文件,生成S001R04等三个npy文件，其余edf文件同理

model_test_new_237.pth：第238轮生成的模型文件，有最佳性能
描述：在main函数中取消加载模型的注释，并将训练轮次设为0，可进行模型测试
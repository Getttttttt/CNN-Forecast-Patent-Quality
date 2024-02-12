sample_data = next(iter(train_loader))
    sample_input = sample_data[1]  # 假设这是输入数据
    sample_input = [Variable(inp.float()) for inp in sample_input]  # 转换为 Variable

    # 确保输入形状与网络期望的形状相匹配
    # ...

    # 使用网络进行一次前向传播
    output = net(sample_input)

    # 可视化计算图
    make_dot(output, params=dict(list(net.named_parameters()) + [('input', sample_input)]))
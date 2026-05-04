def bubble_sort(arr):
    """
    冒泡排序算法
    
    原理：重复遍历要排序的数列，依次比较相邻的两个元素，
    如果顺序错误就交换它们，直到没有需要交换的元素为止。
    
    参数:
        arr: 待排序的列表
    
    返回:
        排序后的列表（原地排序）
    """
    n = len(arr)
    # 外层循环控制遍历轮数
    for i in range(n - 1):
        # 标志位，用于优化：如果某一轮没有交换，说明已经有序
        swapped = False
        # 内层循环进行相邻元素比较，每轮结束会把最大值"冒泡"到最后
        # -i 是因为每一轮结束后，最后i个元素已经排好序
        for j in range(n - 1 - i):
            # 如果前一个元素大于后一个元素，则交换
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果这一轮没有发生交换，说明列表已经有序，提前结束
        if not swapped:
            break
    return arr


# ========== 测试用例 ==========

def test_bubble_sort():
    """测试冒泡排序的各种情况"""
    
    # 测试用例1: 正常乱序
    arr1 = [64, 34, 25, 12, 22, 11, 90]
    print(f"测试1 - 原始数组: {arr1}")
    print(f"      排序结果: {bubble_sort(arr1)}")
    assert arr1 == [11, 12, 22, 25, 34, 64, 90], "测试1失败"
    print("      测试通过 ✅\n")
    
    # 测试用例2: 已排序数组
    arr2 = [1, 2, 3, 4, 5]
    print(f"测试2 - 原始数组: {arr2}")
    print(f"      排序结果: {bubble_sort(arr2)}")
    assert arr2 == [1, 2, 3, 4, 5], "测试2失败"
    print("      测试通过 ✅\n")
    
    # 测试用例3: 逆序数组
    arr3 = [5, 4, 3, 2, 1]
    print(f"测试3 - 原始数组: {arr3}")
    print(f"      排序结果: {bubble_sort(arr3)}")
    assert arr3 == [1, 2, 3, 4, 5], "测试3失败"
    print("      测试通过 ✅\n")
    
    # 测试用例4: 包含重复元素
    arr4 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print(f"测试4 - 原始数组: {arr4}")
    print(f"      排序结果: {bubble_sort(arr4)}")
    assert arr4 == [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9], "测试4失败"
    print("      测试通过 ✅\n")
    
    # 测试用例5: 空数组
    arr5 = []
    print(f"测试5 - 原始数组: {arr5}")
    print(f"      排序结果: {bubble_sort(arr5)}")
    assert arr5 == [], "测试5失败"
    print("      测试通过 ✅\n")
    
    # 测试用例6: 单个元素
    arr6 = [42]
    print(f"测试6 - 原始数组: {arr6}")
    print(f"      排序结果: {bubble_sort(arr6)}")
    assert arr6 == [42], "测试6失败"
    print("      测试通过 ✅\n")
    
    # 测试用例7: 浮点数
    arr7 = [3.14, 2.71, 1.618, 0.618]
    print(f"测试7 - 原始数组: {arr7}")
    print(f"      排序结果: {bubble_sort(arr7)}")
    assert arr7 == [0.618, 1.618, 2.71, 3.14], "测试7失败"
    print("      测试通过 ✅\n")
    
    print("=" * 40)
    print("所有测试用例均通过！🎉")
    print("=" * 40)


if __name__ == "__main__":
    test_bubble_sort()

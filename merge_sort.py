"""
归并排序（Merge Sort）实现
时间复杂度：O(n log n)
空间复杂度：O(n)
稳定性：稳定
"""


def merge_sort(arr):
    """
    归并排序主函数
    
    参数:
        arr: 待排序的列表
    
    返回:
        排序后的列表（升序）
    """
    # 基本情况：如果列表为空或只有一个元素，无需排序
    if len(arr) <= 1:
        return arr
    
    # 1. 分解（Divide）：将数组从中间分成两半
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # 2. 递归求解（Conquer）：递归地对两个子数组进行排序
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)
    
    # 3. 合并（Merge）：将两个已排序的子数组合并为一个有序数组
    return merge(left_sorted, right_sorted)


def merge(left, right):
    """
    合并两个已排序的数组
    
    参数:
        left: 已排序的左半部分列表
        right: 已排序的右半部分列表
    
    返回:
        合并后的有序列表
    """
    result = []      # 存储合并结果的列表
    i = j = 0        # 两个指针，分别指向 left 和 right 的当前元素
    
    # 比较两个数组的元素，将较小的放入结果列表
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将左数组剩余元素加入结果
    result.extend(left[i:])
    
    # 将右数组剩余元素加入结果
    result.extend(right[j:])
    
    return result


def merge_sort_in_place(arr, left=0, right=None):
    """
    原地归并排序（使用辅助数组，但直接在原数组上操作）
    
    参数:
        arr: 待排序的列表（会在原列表上修改）
        left: 左边界索引
        right: 右边界索引（不包含）
    """
    if right is None:
        right = len(arr)
    
    # 如果子数组长度小于等于1，无需排序
    if right - left <= 1:
        return
    
    # 分解
    mid = (left + right) // 2
    
    # 递归排序左右两部分
    merge_sort_in_place(arr, left, mid)
    merge_sort_in_place(arr, mid, right)
    
    # 合并
    _merge_in_place(arr, left, mid, right)


def _merge_in_place(arr, left, mid, right):
    """
    原地合并两个有序子数组 arr[left:mid] 和 arr[mid:right]
    """
    left_part = arr[left:mid]   # 拷贝左半部分
    right_part = arr[mid:right] # 拷贝右半部分
    
    i = j = 0
    k = left
    
    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1
    
    # 处理剩余元素
    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1
    
    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1


# ========== 测试用例 ==========

def test_merge_sort():
    """运行所有测试用例"""
    print("=" * 50)
    print("归并排序测试报告")
    print("=" * 50)
    
    # 测试用例列表：(输入, 期望输出, 测试名称)
    test_cases = [
        ([], [], "空列表"),
        ([5], [5], "单个元素"),
        ([3, 1, 2], [1, 2, 3], "三个元素"),
        ([64, 34, 25, 12, 22, 11, 90], [11, 12, 22, 25, 34, 64, 90], "常规测试"),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5], "逆序数组"),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "已排序数组"),
        ([5, 5, 3, 3, 1, 1], [1, 1, 3, 3, 5, 5], "包含重复元素"),
        ([0, -5, 10, -3, 7], [-5, -3, 0, 7, 10], "包含负数"),
        ([1.5, 3.2, 2.1, 0.8], [0.8, 1.5, 2.1, 3.2], "浮点数"),
    ]
    
    passed = 0
    failed = 0
    
    # 测试 merge_sort（返回新列表版本）
    print("\n▶ 测试 merge_sort（返回新列表）:")
    for input_arr, expected, name in test_cases:
        # 复制输入以免影响其他测试
        test_input = input_arr[:]
        result = merge_sort(test_input)
        if result == expected:
            print(f"  ✓ {name}: {input_arr} -> {result}")
            passed += 1
        else:
            print(f"  ✗ {name}: 输入 {input_arr}, 期望 {expected}, 得到 {result}")
            failed += 1
    
    # 测试 merge_sort_in_place（原地排序版本）
    print("\n▶ 测试 merge_sort_in_place（原地排序）:")
    for input_arr, expected, name in test_cases:
        test_input = input_arr[:]
        merge_sort_in_place(test_input)
        if test_input == expected:
            print(f"  ✓ {name}: {input_arr} -> {test_input}")
            passed += 1
        else:
            print(f"  ✗ {name}: 输入 {input_arr}, 期望 {expected}, 得到 {test_input}")
            failed += 1
    
    # 总结
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"总计: {total} 个测试用例")
    print(f"通过: {passed} ✓")
    print(f"失败: {failed} ✗")
    print(f"通过率: {passed/total*100:.1f}%")
    print('=' * 50)


if __name__ == "__main__":
    # 运行测试
    test_merge_sort()
    
    # 演示示例
    print("\n\n演示示例:")
    example = [38, 27, 43, 3, 9, 82, 10]
    print(f"原始数组: {example}")
    
    sorted_arr = merge_sort(example)
    print(f"排序结果 (返回新列表): {sorted_arr}")
    
    example2 = [38, 27, 43, 3, 9, 82, 10]
    merge_sort_in_place(example2)
    print(f"排序结果 (原地排序):    {example2}")

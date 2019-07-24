import Foundation

class Sort{
    func selectionSort(_ nums: [Int]) -> [Int] {
        var nums = nums
        for i in nums.indices{
            if let min = nums[i...].min(), let iMin = nums.firstIndex(where: { $0 == min }){
                (nums[i], nums[iMin]) = (nums[iMin], nums[i])
            }
        }
        return nums
    }
    
    func insertionSort(_ nums: [Int], hValue: Int = 1) -> [Int]{
        var nums = nums
        for i in nums.indices{
            for j in stride(from: i, to: 0, by: -hValue) {
                if nums[j] < nums[j-hValue] {
                    (nums[j], nums[j-hValue]) = (nums[j-hValue], nums[j])
                }
            }
        }
        return nums
    }
    
    func shellSort(_ nums: [Int]) -> [Int]{
        var nums = nums
        var h = 1
        while (h < 3/nums.count) {
            h = 3 * h + 1
        }
        while h >= 1 {
            nums = insertionSort(nums, hValue: h)
            h = h / 3
        }
        return nums
    }
    
    func test() -> Bool {
        let o = Sort()
        let test = [1, 3, 35, 675, 26, 7, 14, 6, 67, 8431, 56, 21, 78]
        let r_selectionSort = o.selectionSort(test)
        let r_insertionSort = o.insertionSort(test)
        let r_shellSort = o.shellSort(test)
        return r_selectionSort == r_insertionSort && r_shellSort == r_insertionSort
    }
}

import Foundation

public class SortSolution{
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
        while (h < nums.count/3) {
            h = 3 * h + 1
        }
        while h >= 1 {
            nums = insertionSort(nums, hValue: h)
            h = h / 3
        }
        return nums
    }
    
    func mergeSort(_ nums: [Int]) -> [Int]{
        var nums = nums
        divide(&nums, lo: nums.startIndex, hi: nums.endIndex-1)
        return nums
    }
    
    func quickSort(_ nums: [Int]) -> [Int]{
        var nums = nums
        quick(&nums, lo: nums.startIndex, hi: nums.endIndex-1)
        return nums
    }
    
    public static func test() -> Bool {
        let o = SortSolution()
        let test = [1, 3, 35, 675, 26, 7, 14, 6, 67, 8431, 56, 21, 78]
        let r_selectionSort = o.selectionSort(test)
        let r_insertionSort = o.insertionSort(test)
        let r_shellSort = o.shellSort(test)
        let r_mergeSort = o.mergeSort(test)
        let r_quickSort = o.quickSort(test)
        return r_selectionSort == r_insertionSort
            && r_shellSort == r_insertionSort
            && r_mergeSort == r_insertionSort
            && r_quickSort == r_insertionSort
    }
}

//MARK:- QuickSort
private extension SortSolution{
    func quick(_ nums: inout [Int], lo: Int, hi: Int){
        if hi <= lo { return }
        let j = partition(&nums, lo, hi)
        quick(&nums, lo:  lo, hi: j-1)
        quick(&nums, lo: j+1, hi: hi)
    }
    
    func partition<T: Comparable>(_ a: inout [T], _ lo: Int, _ hi: Int) -> Int {
        let pivot = a[lo]
        var i = lo
        var j = hi + 1
        
        while true {
            repeat { i += 1 } while i != hi && a[i] < pivot
            repeat { j -= 1 } while j != lo && a[j] > pivot
            if i < j {
                (a[i], a[j]) = (a[j], a[i])
            } else {
                break
            }
        }
        (a[lo], a[j]) = (a[j], a[lo])
        return j
    }
}

//MARK:- MergeSort
private extension SortSolution{
    func merge(_ a: inout [Int], lo: Int, hi: Int){
        let copied = a[lo...hi]
        let c_lo = copied.startIndex
        let c_hi = copied.endIndex - 1
        let c_mid = (c_lo + c_hi) / 2
        var i = c_lo, j = c_mid + 1
        
        for k in lo...hi{
            if (i > c_mid) {
                a[k] = copied[j]
                j += 1
            }
            else if (j > c_hi) {
                a[k] = copied[i]
                i += 1
            }
            else if copied[j] < copied[i] {
                a[k] = copied[j]
                j += 1
            }
            else {
                a[k] = copied[i]
                i += 1
            }
        }
    }
    
    func divide(_ nums: inout [Int], lo: Int, hi: Int){
        if hi <= lo { return }
        let mid = (lo + hi) / 2
        divide(&nums, lo: lo, hi: mid)
        divide(&nums, lo: mid+1, hi: hi)
        merge(&nums, lo: lo, hi: hi)
    }
}

//MARK:- Heap
extension SortSolution{
    struct Heap<T> {
        typealias Priority = (T, T) -> Bool
        
        fileprivate typealias Index = Int
        
        var elements: [T]
        
        private let priority: Priority
        
        init(elements: [T] = [], priority: @escaping Priority) {
            self.elements = elements
            self.priority = priority
            buildHeap()
        }
        
        private mutating func buildHeap() {
            for index in (0 ..< count / 2).reversed() {
                siftDown(index)
            }
        }
        
        var isEmpty: Bool { return elements.isEmpty }
        
        var count: Int { return elements.count }
        
        var peek: T? { return elements.first }
        
        mutating func push(_ element: T) {
            elements.append(element)
            siftUp(elements.endIndex-1)
        }
        
        mutating func pop() -> T? {
            guard !isEmpty else { return nil }
            swapElement(at: elements.startIndex, with: elements.endIndex-1)
            let element = elements.removeLast()
            siftDown(elements.startIndex)
            return element
        }
        
        private mutating func siftUp(_ index: Index) {
            guard !index.isRoot else { return }
            guard isHigherPriority(at: index, than: index.parent) else { return }
            swapElement(at: index, with: index.parent)
            siftUp(index.parent)
        }
        
        private mutating func siftDown(_ index: Index) {
            guard let nextIndex = getHigherPriority(from: index) else { return }
            swapElement(at: index, with: nextIndex)
            siftDown(nextIndex)
        }
        
        private func isHigherPriority(at first: Index, than second: Index) -> Bool {
            return priority(elements[first], elements[second])
        }
        
        private func highestPriorityIndex(of parent: Index, and child: Index) -> Index {
            guard child < count else { return parent }
            return isHigherPriority(at: child, than: parent) ? child : parent
        }
        
        private func getHigherPriority(from parent: Index) -> Index? {
            let ifLeft  = highestPriorityIndex(of: parent, and: parent.leftChild)
            let ifRight = highestPriorityIndex(of: ifLeft, and: parent.rightChild)
            return parent == ifRight ? nil : ifRight
        }
        
        private mutating func swapElement(at firstIndex: Int, with secondIndex: Int) {
            guard firstIndex != secondIndex else { return }
            elements.swapAt(firstIndex, secondIndex)
        }
        
        public mutating func sort() -> [T] {
            var result: [T] = []
            while count > 0, let e = pop() {
                result.append(e)
            }
            return result
        }
    }
}
private extension SortSolution.Heap.Index{
    
    var isRoot: Bool { return self == 0 }
    
    var leftChild: Int { return (2 * self) + 1 }
    
    var rightChild: Int { return (2 * self) + 2 }
    
    var parent: Int { return (self - 1) / 2 }
}


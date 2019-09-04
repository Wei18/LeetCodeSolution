//
//  Solution+Easy.swift
//
//
//  Created by Wei on 2019/02/19.
//

import Foundation

public class Solution {
    public init(){}
}

//MARK:- Easy
public extension Solution{
    
    /**
     1. Two Sum
     */
    func twoSum1(_ nums: [Int], _ target: Int) -> [Int] {
        let nums = nums
        var res: [Int] = []
        var dict:[Int:Int] = [:]
        nums.enumerated().forEach { (index, value) in
            if let found = dict[target-value] {
                res.append(index)
                res.append(found)
                return
            }
            else{
                dict[value] = index
            }
        }
        return res
    }
    
    /**
     7. Reverse Integer
     */
    func reverse(_ x: Int) -> Int {
        var res = 0
        var x = x
        repeat{
            res = res * 10 + x % 10
            x /= 10
        }while x != 0
        return (res > Int32.max || res < Int32.min) ? 0 : res
    }
    
    /**
     13. Roman to Integer
     */
    func romanToInt(_ s: String) -> Int {
        let roman = [
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000]
        
        var res = 0
        var i = 0
        
        func get(_ i: String.Index) -> Int{
            return roman[s[i].description]!
        }
        
        for (i,_) in s.enumerated(){
            guard i + 1 < s.count else { break }
            let curI = s.index(s.startIndex, offsetBy: i)
            let nextI = s.index(s.startIndex, offsetBy: i + 1)
            let curVal = get(curI)
            if curVal < get(nextI) {
                res -= curVal
            }else{
                res += curVal
            }
        }
        
        let lastI = s.index(before: s.endIndex)
        res += get(lastI)
        return res
    }
    
    /**
     20. Valid Parentheses
     */
    func isValid(_ s: String) -> Bool {
        var leftBraces: [Character] = []
        for i in s.indices{
            switch s[i] {
            case "(", "[", "{":
                leftBraces.append(s[i])
            default:
                let rightBrace = s[i]
                guard let leftBrace  = leftBraces.popLast() else { return false }
                let match = "\(leftBrace)\(rightBrace)"
                if !"()[]{}".contains(match) {
                    return false
                }
            }
        }
        return leftBraces.isEmpty
    }

    /**
     21. Merge Two Sorted Lists
     */
    func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        guard let l1 = l1 else { return l2 }
        guard let l2 = l2 else { return l1 }
        
        var head: ListNode
        if l1.val < l2.val{
            head = l1
            head.next = mergeTwoLists(head.next, l2)
        }else{
            head = l2
            head.next = mergeTwoLists(head.next, l1)
        }
        return head
    }

    /**
     26. Remove Duplicates from Sorted Array
     */
    func removeDuplicates(_ nums: inout [Int]) -> Int {
        guard nums.count > 1 else { return nums.count }
        var count = 1
        for i in 1..<nums.endIndex where nums[i] != nums[i-1] {
            nums[count] = nums[i]
            count += 1
        }
        return count
    }
    
    
    /**
     28. Implement strStr()
     */
    func strStr(_ haystack: String, _ needle: String) -> Int {
        /*
         edge
         return non
         init
         compute index of haystack
         if first character of haystck is equal to first of needle
         then loop check next until count of needle
         return -1 whil index of haystack reach endindex.
         */
        
        let haystack = ArraySlice(haystack)
        let needle = ArraySlice(needle)
        guard !needle.isEmpty else { return 0 }
        guard haystack.count >= needle.count else { return -1 }
        
        var i = haystack.startIndex
        while i < haystack.endIndex-needle.endIndex+1 {
            let subHay = haystack[i..<(i+needle.count)]
            if subHay == needle {
                return i
            }
            i += 1
        }
        
        return -1
    }

    /**
     35. Search Insert Position
     */
    func searchInsert(_ nums: [Int], _ target: Int) -> Int {
        return nums.firstIndex{ $0 >= target } ?? nums.count
    }
    
    
    /**
     53. Maximum Subarray
     */
    func maxSubArray(_ nums: [Int]) -> Int {
        func kanade(_ nums: [Int]) -> Int{
            guard let first = nums.first else { return 0 }
            var maxSoFar = first
            var maxEndingHere = first
            for i in 1..<nums.count {
                maxEndingHere = max(nums[i], maxEndingHere + nums[i])
                maxSoFar = max(maxSoFar, maxEndingHere)
            }
            return maxSoFar
        }
        return kanade(nums)
    }
    

    
    /**
     58. Length of Last Word
     */
    func lengthOfLastWord(_ s: String) -> Int {
        return s.split(separator: " ").last?.count ?? 0
    }
    
    
    /**
     69. Sqrt(x)
     */
    func mySqrt(_ x: Int) -> Int {
        guard x > 0 else { return 0 }
        /*
         recursively binary search left, right*/
        return recursivelyMySqrt(0, x, x)
    }
    
    func recursivelyMySqrt(_ l: Int, _ r: Int, _ target: Int) -> Int{
        guard l <= r else { return r }
        let mid = (l + r) / 2
        let sqrt = mid * mid
        if sqrt > target {
            return recursivelyMySqrt(l, mid-1, target)
        }
        else if sqrt < target {
            return recursivelyMySqrt(mid+1, r, target)
        }
        else {
            return mid
        }
    }

    /**
     88. Merge Sorted Array
     */
    func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        var j_end = n - 1
        var i_end = m - 1
        
        for i in stride(from: nums1.count - 1, to: -1, by: -1){
            if j_end < 0 {
                nums1[i] = nums1[i_end]
                i_end -= 1
            }else if i_end < 0 {
                nums1[i] = nums2[j_end]
                j_end -= 1
            }else if nums1[i_end] >= nums2[j_end]{
                nums1[i] = nums1[i_end]
                i_end -= 1
            }else{ //nums1[i_end] < nums2[j_end]
                nums1[i] = nums2[j_end]
                j_end -= 1
            }
        }
    }

    /**
     100. Same Tree
     */
    func isSameTree(_ p: TreeNode?, _ q: TreeNode?) -> Bool {
        func isLeaf(_ node: TreeNode) -> Bool {
            return node.left == nil && node.right == nil
        }
        
        if p == nil, q == nil{
            return true
        }
        else if let p = p, let q = q {
            if p.val == q.val {
                return isSameTree(p.left, q.left) && isSameTree(p.right, q.right)
            }
            else {
                return false
            }
        }
        else{
            return false
        }
    }

    
    /**
     66. Plus One
     */
    func plusOne(_ digits: [Int]) -> [Int] {
        var nums = digits
        
        nums[nums.count - 1] += 1
        
        for i in stride(from: nums.count-1, to: 0, by: -1){
            if nums[i] == 10, i > 0{
                nums[i] = 0
                nums[i-1] += 1
            }
        }
        
        if nums[0] == 10{
            nums[0] = 0
            nums.insert(1, at: 0)
        }
        
        return nums
    }

    
    /**
     83. Remove Duplicates from Sorted List
     */
    func deleteDuplicates(_ head: ListNode?) -> ListNode? {
        var node = head
        
        while let found = node?.next {
            if node?.val == found.val{
                node?.next = found.next
            }else{
                node = found
            }
        }
        
        return head
    }

    
    /**
     101. Symmetric Tree
     */
    func isSymmetric(_ root: TreeNode?) -> Bool {
        if Bool.random(){
            guard let root = root else { return true }
            return DFSIsSymmetric(root.left, root.right)
        }else{
            return BFSIsSymmetric(root)
        }
    }
    
    func DFSIsSymmetric(_ left: TreeNode?, _ right: TreeNode?) -> Bool {
        guard let l = left, let r = right else {
            /*
             both are nil will return true, otherwise return false
             */
            return left === right
        }
        
        if l.val != r.val {
            return false
        }else{
            return DFSIsSymmetric(l.left, r.right)
                && DFSIsSymmetric(l.right, r.left)
        }
    }
    
    func BFSIsSymmetric(_ root: TreeNode?) -> Bool {
        guard let root = root else { return true }
        
        var qLeft: [TreeNode?] = [root.left]
        var qRight: [TreeNode?] = [root.right]
        
        while !qLeft.isEmpty && !qRight.isEmpty {
            let oLeft = qLeft.removeFirst()
            let oRight = qRight.removeFirst()
            
            if oLeft === oRight{
                continue
            }
            
            guard
                let left = oLeft,
                let right = oRight,
                left.val == right.val
                else { return false }
            
            qLeft.append(left.left)
            qLeft.append(left.right)
            qRight.append(right.right)
            qRight.append(right.left)
        }
        return true
    }

    /**
     104. Maximum Depth of Binary Tree
     */
    func maxDepth(_ root: TreeNode?) -> Int {
        return recursivelyMaxDepth(root, count: 0)
    }
    
    func recursivelyMaxDepth(_ root: TreeNode?, count: Int) -> Int{
        guard let node = root else { return count }
        let currentCount = count + 1
        
        let right = recursivelyMaxDepth(node.right, count: currentCount)
        
        let left = recursivelyMaxDepth(node.left, count: currentCount)
        
        return max(right, left, currentCount)
    }
    
    
    /**
     107. Binary Tree Level Order Traversal II
     */
    func levelOrderBottom(_ root: TreeNode?) -> [[Int]] {
        var result: [[Int]] = []
        
        func bfs(){
            guard let node = root else { return }
            var queue = [node]
            while !queue.isEmpty{
                result.append([])
                for _ in queue.indices {
                    let q = queue.removeFirst()
                    result[result.endIndex-1].append(q.val)
                    if let l = q.left  { queue.append(l) }
                    if let r = q.right { queue.append(r) }
                }
            }
        }
        
        func dfs(_ root: TreeNode?, _ floor: Int, _ res: inout [[Int]]){
            guard let node = root else { return }
            
            if floor < res.count {
                res[floor].append(node.val)
            }else{
                res.append([node.val])
            }
            
            dfs(node.left, floor + 1, &res)
            dfs(node.right, floor + 1, &res)
        }
        
        if Bool.random(){
            dfs(root, 0, &result)
        }else{
            bfs()
        }
        return result.reversed()
    }
    
    /**
     111. Minimum Depth of Binary Tree
     */
    func minDepth(_ root: TreeNode?) -> Int {
        return recursivelyMinDepth(root, count: 0) ?? 0
    }
    
    func recursivelyMinDepth(_ root: TreeNode?, count: Int) -> Int? {
        guard let node = root else { return nil }
        let currentCount = count + 1
        
        let right = recursivelyMinDepth(node.right, count: currentCount)
        
        let left = recursivelyMinDepth(node.left, count: currentCount)
        
        if let l = left, let r = right {
            return min(l, r)
        }
        else if let l = left{
            return l
        }
        else if let r = right {
            return r
        }
        else {
            return currentCount
        }
    }
    
    
    /**
     112. Path Sum
     */
    func hasPathSum(_ root: TreeNode?, _ sum: Int) -> Bool {
        guard let node = root else { return false }
        let currentValue = sum - node.val
        if currentValue == 0, node.isLeaf {
            return true
        }
        if hasPathSum(node.left, currentValue) {
            return true
        }
        if hasPathSum(node.right, currentValue) {
            return true
        }
        return false
    }
    
    
    /**
     118. Pascal's Triangle
     */
    func generate(_ numRows: Int) -> [[Int]] {
        //initial
        var dp: [[Int]] = Array(repeating: [], count: numRows)
        
        for i in dp.indices{
            dp[i] = Array(repeating: 1, count: i+1)
        }
        
        //edge condition
        guard dp.count >= 2 else { return dp }
        
        //compute
        for i in 2..<dp.endIndex{
            for j in 1..<dp[i].endIndex-1{
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
            }
        }
        
        return dp
    }


    /**
     136. Single Number
     */
    func singleNumber(_ nums: [Int]) -> Int {
        /*
         edge: count of arr of nums > 0
         using XOR (^)
         definition XOR
         == return 0
         != return 1
         */
        
        guard !nums.isEmpty else { return 0 }
        var single = nums.first!
        for i in 1..<nums.endIndex {
            single ^= nums[i]
        }
        return single
        //using swift
        //return nums.reduce(0, ^)
    }

    /**
     167. Two Sum II - Input array is sorted
     */
    func twoSum167(_ numbers: [Int], _ target: Int) -> [Int] {
        var i = numbers.startIndex
        var j = numbers.endIndex - 1
        
        while i < j {
            let sum = numbers[i] + numbers[j]
            if sum < target {
                i += 1
            } else if target < sum {
                j -= 1
            } else {
                return [i + 1, j + 1]
            }
        }
        
        return []
    }
    
    
    /**
     169. Majority Element
     */
    func majorityElement(_ nums: [Int]) -> Int {
        var dict: [Int: Int] = [:]
        var currentMaxValue = -1
        dict[currentMaxValue] = 0
        
        for i in nums.indices{
            dict[nums[i], default: 0] += 1
            
            if dict[nums[i]]! > dict[currentMaxValue]! {
                currentMaxValue = nums[i]
            }
        }
        
        return currentMaxValue
    }

    /**
     202. Happy Number
     */
    func isHappy(_ n: Int) -> Bool {
        var record = [Int: Int]()
        return sumOfSquares(n, set: &record) == 1
    }
    
    func sumOfSquares(_ n: Int, set: inout [Int: Int]) -> Int {
        set[n] = 1
        var n = n
        var newN = 0
        while n > 0 {
            let modNum = n % 10
            newN = newN + modNum * modNum
            n /= 10
        }
        
        if set[newN] != nil {
            return newN
        }else{
            return sumOfSquares(newN, set: &set)
        }
    }

    
    /**
     206. Reverse Linked List
     */
    func reverseList(_ head: ListNode?) -> ListNode? {
        return iterativelyReverseList(head)
    }
    
    func iterativelyReverseList(_ head: ListNode?) -> ListNode? {
        var current: ListNode? = head
        var previous: ListNode? = nil
        
        while current != nil {
            /* swap 3 listnode
             example:
             head: 1 -> 2 -> 3 -> 4
             keep        :   2, 3, 4, nil,
             current.next: nil, 1, 2,   3,
             previous    :   1, 2, 3,   4,
             current     :   2, 3, 4, nil,
             */
            let keep = current?.next
            current?.next = previous
            previous = current
            current = keep
        }
        
        return previous
    }

    
    /**
     217. Contains Duplicate
     */
    func containsDuplicate(_ nums: [Int]) -> Bool {
        var dict: [Int: Int] = [:]
        
        for i in nums.indices {
            dict[nums[i], default: 0] += 1
            if dict[nums[i]]! > 1 {
                return true
            }
        }
        return false
    }

    /**
     225. Implement Stack using Queues
     */
    class MyStack {
        
        private var items: [Int] = []
        /** Initialize your data structure here. */
        init() {
            
        }
        
        /** Push element x onto stack. */
        func push(_ x: Int) {
            items.append(x)
        }
        
        /** Removes the element on top of the stack and returns that element. */
        func pop() -> Int {
            return items.removeLast()
        }
        
        /** Get the top element. */
        func top() -> Int {
            return items.last!
        }
        
        /** Returns whether the stack is empty. */
        func empty() -> Bool {
            return items.isEmpty
        }
    }

    
    /**
     226. Invert Binary Tree
     */
    func invertTree(_ root: TreeNode?) -> TreeNode? {
        if Bool.random(){
            //bfs
            guard let node = root else { return nil }
            var queue = [node]
            
            while !queue.isEmpty{
                let q = queue.removeFirst()
                (q.left, q.right) = (q.right, q.left)
                
                if let l = q.left{
                    queue.append(l)
                }
                if let r = q.right{
                    queue.append(r)
                }
            }
            return node
        }else{
            //dfs
            return dfsInvertTree(root)
        }
    }
    
    func dfsInvertTree(_ root: TreeNode?) -> TreeNode?{
        //postorder
        guard let node = root else { return nil }
        let l = dfsInvertTree(node.left)
        let r = dfsInvertTree(node.right)
        node.left = r
        node.right = l
        return node
    }

    /**
     232. Implement Queue using Stacks
     */
    class MyQueue {
        private var items: [Int] = []
        private var dequeueIndex = 0
        private var euqueueIndex: Int { return items.count - 1 }
        
        /** Initialize your data structure here. */
        init() {
            
        }
        
        /** Push element x to the back of queue. */
        func push(_ x: Int) {
            items.append(x)
        }
        
        /** Removes the element from in front of queue and returns that element. */
        func pop() -> Int {
            dequeueIndex += 1
            return items[dequeueIndex - 1]
        }
        
        /** Get the front element. */
        func peek() -> Int {
            return items[dequeueIndex]
        }
        
        /** Returns whether the queue is empty. */
        func empty() -> Bool {
            return dequeueIndex - 1 == euqueueIndex
        }
    }

    
    /**
     234. Palindrome Linked List
     */
    func isPalindrome(_ head: ListNode?) -> Bool {
        /*
         required: O(n) time and O(1) space
         get middle of nodes
         reverse nodes from the middle to the end.
         compare both headList and reversedList
         return falst while unexpected until the middle then return true
         */
        func getMiddle(_ head: ListNode?) -> ListNode? {
            var slow = head
            var fast = head?.next
            while let found = fast?.next?.next {
                slow = slow?.next //1, 2 3 4
                fast = found      //2, 4 6 8
            }
            if let _ = fast?.next {
                //for count of nodes is odd
                slow = slow?.next
            }
            return slow
        }
        
        func reverse(_ head: ListNode?) -> ListNode?{
            guard let nextNode = head?.next else { return head }
            let node = reverse(nextNode)
            head?.next?.next = head
            head?.next = nil
            return node
        }
        
        let middle = getMiddle(head)
        let reversed = reverse(middle)
        middle?.next = nil
        
        var nodes: (ListNode?, ListNode?) = (head, reversed)
        while let no0 = nodes.0, let no1 = nodes.1 {
            if no0.val != no1.val {
                return false
            }
            nodes.0 = no0.next
            nodes.1 = no1.next
        }
        return true
    }

    /**
     268. Missing Number
     */
    func missingNumber(_ nums: [Int]) -> Int {
        return nums.count * (nums.count + 1) / 2 - nums.reduce(0){ $0 + $1 }
    }
    
    
    /**
     283. Move Zeroes
     */
    func moveZeroes(_ nums: inout [Int]) {
        var i = 0
        
        for k in nums.indices{
            if nums[k] != 0{
                nums[i] = nums[k]
                i += 1
            }
        }
        
        for j in i..<nums.endIndex{
            nums[j] = 0
        }
    }

    /**
     344. Reverse String
     */
    func reverseString(_ s: String) -> String {
        return String(s.reversed())
    }
    
    /**
     350. Intersection of Two Arrays II
     */
    func intersect(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        /*
         sorted nums1
         using binary search
         check elements of num2
         */
        
        var nums1 = nums1.sorted()
        func searchPop(_ nums: inout [Int], target: Int) -> Int?{
            var lo = nums.startIndex
            var hi = nums.endIndex
            while lo < hi {
                let mid = (lo + hi) >> 1
                if nums[mid] > target {
                    hi -= 1
                }else if nums[mid] < target{
                    lo += 1
                }else{
                    return nums.remove(at: mid)
                }
            }
            return nil
        }
        
        var res: [Int] = []
        for i in nums2.indices{
            if let t = searchPop(&nums1, target: nums2[i]){
                res.append(t)
            }
        }
        return res
    }

    /**
     371. Sum of Two Integers
     */
    func getSum(_ a: Int, _ b: Int) -> Int {
        /*pf
         0 + 0 = 00
         1 + 0 = 10
         0 + 1 = 01
         1 + 1 = 10 //carry
         
         x0 = a (xor) b
         c0 = (a & b) << 1 //carry
         a + b = x0 + c0
         
         if has x1 = x0 (xor) c0, c1 = (x0 & c0) << 1
         that mean x0 + c0 = x1 + c1
         while cN == 0, a + b = xN + cN = xN
         */
        if b == 0 { return a }
        let xor = a ^ b
        let and = a & b
        let carry = and << 1
        return getSum(xor, carry)
    }

    /**
     412. Fizz Buzz
     */
    enum FizzBuzz: String{
        case Fizz
        case Buzz
        case FizzBuzz
    }
    func fizzBuzz(_ n: Int) -> [String] {
        return (1...n).map{ num -> String in
            switch num{
            case _ where num % 15 == 0:
                return FizzBuzz.FizzBuzz.rawValue
            case _ where num % 3 == 0:
                return FizzBuzz.Fizz.rawValue
            case _ where num % 5 == 0:
                return FizzBuzz.Buzz.rawValue
            default:
                return String(num)
            }
        }
    }
    
    
    /**
     414. Third Maximum Number
     */
    func thirdMax(_ nums: [Int]) -> Int {
        let r = nums.reduce(into: (Int.min, Int.min, Int.min)) { (r, v) in
            switch v {
            case _ where v == r.0 || v == r.1 || v == r.2:
                break
            case _ where v > r.0:
                r = (v, r.0, r.1)
            case _ where v > r.1:
                (r.1, r.2) = (v, r.1)
            case _ where v > r.2:
                r.2 = v
            default:
                break
            }
        }
        return (r.2 == Int.min) ? r.0 : r.2
    }
    
    
    /**
     438. Find All Anagrams in a String
     */
    func findAnagrams(_ s: String, _ p: String) -> [Int] {
        var result = [Int]()
        var ref = Array<Int>(repeating: 0, count: 26)
        var map = ref
        
        let p = p.unicodeScalars
            .map{ $0.value - Unicode.Scalar("a").value }
            .map{ Int($0) }
        let s = s.unicodeScalars
            .map{ $0.value - Unicode.Scalar("a").value }
            .map{ Int($0) }
        
        p.forEach{ ref[$0] += 1 }
        
        for i in 0..<s.count {
            map[s[i]] += 1
            if i - p.count >= 0 {
                map[s[i-p.count]] -= 1
            }
            
            if map == ref {
                result.append(i - p.count + 1)
            }
        }
        return result
    }


    /**
     461. Hamming Distance
     */
    func hammingDistance(_ x: Int, _ y: Int) -> Int {
        return (x ^ y).nonzeroBitCount
    }

    /**
     447. Number of Boomerangs
     */
    func numberOfBoomerangs(_ points: [[Int]]) -> Int {
        func getDistance(_ p0: [Int], _ p1: [Int]) -> Int {
            let a = p0[0] - p1[0]
            let b = p0[1] - p1[1]
            return a * a + b * b
        }
        
        var res = 0
        for i in points.indices {
            var dict: [Int: Int] = [:]
            
            for j in points.indices {
                guard i != j else { continue }
                let distance = getDistance(points[i], points[j])
                dict[distance, default: 0] += 1
            }
            
            for val in dict.values {
                //Combination
                //from the number of n elements
                //get the number of k elements.
                // n!/(k!)(n-k)!
                // while k == 2, n * (n-1) / 2
                // each distance > 2 can get 2 result
                if val >= 2{
                    res += val * (val - 1) /* / 2 * 2 */
                }
            }
        }
        
        return res
    }

    /**
     476. Number Complement
     */
    func findComplement(_ num: Int) -> Int {
        var mask = ~0
        while num & mask > 0{
            mask <<= 1
        }
        return ~mask & ~num
    }

    
    /**
     500. Keyboard Row
     */
    enum Row: String{
        case one   = "qwertyuiop"
        case two   = "asdfghjkl"
        case third = "zxcvbnm"
    }
    func findWords(_ words: [String]) -> [String] {
        return words.compactMap{
            var isChanged = false
            var row: Row?{
                didSet{
                    guard oldValue != nil, oldValue != row else { return }
                    isChanged = true
                }
            }
            for letter in $0.lowercased(){
                if Row.one.rawValue.contains(letter){
                    row = .one
                }else if Row.two.rawValue.contains(letter){
                    row = .two
                }else if Row.third.rawValue.contains(letter){
                    row = .third
                }else{}
                
                if isChanged{
                    return nil
                }else{
                    continue
                }
            }
            return $0
        }
    }
    
    
    /**
     509. Fibonacci Number
     */
    func fib(_ N: Int) -> Int {
        var res = Array(repeating: 0, count: N+1)
        DPFib(N, res: &res)
        return res[N]
    }
    func DPFib(_ N: Int, res dp: inout [Int]) {
        if N < 2 {
            dp[N] = N
        }else if dp[N] == 0 {
            DPFib(N-1, res: &dp)
            DPFib(N-2, res: &dp)
            dp[N] = dp[N-1] + dp[N-2]
        }else{
            
        }
    }

    /**
     530. Minimum Absolute Difference in BST
     */
    func getMinimumDifference(_ root: TreeNode?) -> Int {
        /*
         use inorder traversal in depth first search
         to get sorted values(<) of each node from the BST
         and then keep previous to calculated minimum result
         */
        func inorder(_ root: TreeNode?, _ prev: inout TreeNode?) -> Int {
            guard let node = root else { return Int.max }
            var result = inorder(node.left, &prev)
            if let prev = prev {
                result = min(result, node.val-prev.val)
            }
            prev = node
            let rR = inorder(node.right, &prev)
            return min(result, rR)
        }
        var captured: TreeNode?
        return inorder(root, &captured)
    }

    /**
     538. Convert BST to Greater Tree
     */
    func convertBST(_ root: TreeNode?) -> TreeNode? {
        var sum = 0
        return recursivelyGreaterTree(root, sum: &sum)
    }
    
    func recursivelyGreaterTree(_ root: TreeNode?, sum: inout Int) -> TreeNode?{
        guard let node = root else { return nil }
        _ = recursivelyGreaterTree(node.right, sum: &sum)
        node.val += sum
        sum = node.val
        _ = recursivelyGreaterTree(node.left, sum: &sum)
        return node
    }
    
    /**
     541. Reverse String II
     */
    func reverseStr(_ s: String, _ k: Int) -> String {
        func reverse(_ arr: inout [Character], start: Int, end: Int){
            var l = start, r = end
            let mid = (l + r) / 2
            while l <= mid {
                (arr[l], arr[r]) = (arr[r], arr[l])
                l += 1
                r -= 1
            }
        }
        
        
        var s: [Character] = Array(s)
        
        var idx = s.startIndex
        
        while idx < s.endIndex {
            let endIndex = min(idx + k, s.endIndex) - 1
            reverse(&s, start: idx, end: endIndex)
            idx += 2 * k
        }
        
        return String(s)
    }

    /**
     557. Reverse Words in a String III
     */
    func reverseWords(_ s: String) -> String {
        return s.split(separator: " ").map{ String($0.reversed()) }.joined(separator: " ")
    }


    /**
     561. Array Partition I
     */
    func arrayPairSum(_ nums: [Int]) -> Int {
        var result = 0
        for (index, value) in nums.sorted(by: <).enumerated(){
            guard index % 2 == 0 else { continue }
            result += value
        }
        return result
    }

    
    /**
     563. Binary Tree Tilt
     */
    func tilting(_ root: TreeNode?, _ tilt: inout Int) -> Int {
        guard let node = root else { return 0 }
        
        let left = tilting(node.left, &tilt)
        let right = tilting(node.right, &tilt)
        tilt += abs(left-right)
        
        return left + right + node.val
    }
    
    func findTilt(_ root: TreeNode?) -> Int {
        guard root != nil else { return 0 }
        var tilt = 0
        _ = tilting(root, &tilt)
        return tilt
    }
    
    
    /**
     566. Reshape the Matrix
     */
    func matrixReshape(_ nums: [[Int]], _ r: Int, _ c: Int) -> [[Int]] {
        let old_r = nums.count
        let old_c = nums.first?.count ?? 0
        let total = old_c * old_r
        guard total == r * c else { return nums }
        
        let resC = Array(repeating: 0, count: c)
        var resR = Array(repeating: resC, count: r)
        
        for i in 0..<total{
            resR[i/c][i%c] = nums[i/old_c][i%old_c]
        }
        
        return resR
    }

    /**
     617. Merge Two Binary Trees
     */
    func mergeTrees(_ t1: TreeNode?, _ t2: TreeNode?) -> TreeNode? {
        if t1 == nil {
            return t2
        }else{
            recursivelyMergeTrees(t1, t2)
            return t1
        }
    }
    
    func recursivelyMergeTrees(_ t1: TreeNode?, _ t2: TreeNode?){
        t1?.val += t2?.val ?? 0
        
        if t1?.left != nil || t2?.left != nil{
            if t1?.left == nil{
                t1?.left = TreeNode(0)
            }
            recursivelyMergeTrees(t1?.left, t2?.left)
        }
        if t1?.right != nil || t2?.right != nil{
            if t1?.right == nil{
                t1?.right = TreeNode(0)
            }
            recursivelyMergeTrees(t1?.right, t2?.right)
        }
    }

    
    /**
     628. Maximum Product of Three Numbers
     */
    func maximumProduct(_ nums: [Int]) -> Int {
        /*
         max
         ( 1st max * 2nd max * 3rd max )
         ( 1st min * 2nd min * 1st max )
         */
        
        var maxs = [Int.min, Int.min, Int.min]
        var mins = [Int.max, Int.max]
        
        for num in nums {
            if maxs[0] < num {
                maxs[0] = num
                maxs.sort(by: <)
            }
            if mins[0] > num {
                mins[0] = num
                mins.sort(by: >)
            }
        }
        let ans0 = maxs[0] * maxs[1] * maxs[2]
        let ans1 = maxs[2] * mins[0] * mins[1]
        return max(ans0, ans1)
    }

    /**
     633. Sum of Square Numbers
     */
    func judgeSquareSum(_ c: Int) -> Bool {
        /*
         if a^2 + b^2 = c return true
         so we need find a, b
         make a in loop 0..<sqrt(c)
         so we can detect b is satisfy requirements
         
         try to write binary search, two pointers, set as ref:
         https://leetcode.com/problems/sum-of-square-numbers/discuss/104948/Swift-solution-Binary-Search-Set-Two-Pointers
         */
        
        func twePointers(_ c: Int) -> Bool{
            //left, right
            //Note that left += 1, right -= 1.
            let len = Int(sqrt(Double(c)))
            var left = 0
            var right = len
            while left <= right {
                let sum = left * left + right * right
                if sum > c {
                    right -= 1
                }else if sum < c {
                    left += 1
                }else{
                    return true
                }
            }
            return false
        }
        
        func setway(_ c: Int) -> Bool{
            let len = Int(sqrt(Double(c)))
            var aSet = Set<Int>()
            for i in 0...len{
                aSet.insert(i * i)
                if aSet.contains(c - i * i){
                    return true
                }
            }
            return false
        }
        
        switch (1...2).randomElement() {
        case 1:
            return twePointers(c)
        case 2:
            return setway(c)
        default:
            fatalError()
        }
    }

    
    /**
     645. Set Mismatch
     */
    func findErrorNums(_ nums: [Int]) -> [Int] {
        /*
         sum of nums can be calculated by width * height / 2
         use dictionary find the duplicated value
         use reminder + duplicated
         */
        
        var duplicated: Int!
        let sum = (1 + nums.count) * nums.count / 2
        let reminder = nums.reduce(sum, -)
        
        var dict: [Int: Int] = [:]
        for number in nums {
            dict[number, default: 0] += 1
            if dict[number]! > 1 {
                duplicated = number
                break
            }
        }
        
        return [duplicated, reminder+duplicated]
    }

    /**
     657. Robot Return to Origin
     */
    func judgeCircle(_ moves: String) -> Bool {
        var point: (Int, Int) = (0, 0)
        moves.forEach{
            switch $0{
            case "U":
                point.0 += 1
            case "D":
                point.0 -= 1
            case "L":
                point.1 -= 1
            case "R":
                point.1 += 1
            default:
                break
            }
        }
        return point == (0, 0)
    }
    
    /**
     671. Second Minimum Node In a Binary Tree
     */
    func findSecondMinimumValue(_ root: TreeNode?) -> Int {
        /*
         Two conditions:
         Each node in this tree has exactly two or zero sub-node.
         root.val = min(root.left.val, root.right.val)
         
         Round1:
         just check two values between left and right nodes
         if these values are same return -1, if no nodes return -1
         
         Round2
         go through the node that value of node is equal to the value of parent.
         record other value of node that is not equal.
         recurrence record value with method min.
         */
        
        func dfs(_ root: TreeNode?) -> Int {
            guard let node = root  else { return Int.max }
            guard let left = node.left, let right = node.right else { return Int.max }
            
            var result = Int.max
            if node.val == left.val{
                result = dfs(left)
            }else{
                result = left.val
            }
            
            if node.val == right.val{
                result = min(result, dfs(right))
            }else{
                result = min(result, right.val)
            }
            
            return result
        }
        
        let r = dfs(root)
        return r == Int.max ? -1 : r
    }

    /**
     682. Baseball Game
     */
    func calPoints(_ ops: [String]) -> Int {
        var sum: [Int] = []
        ops.forEach{ letter in
            switch letter {
            case "C":
                _ = sum.popLast()
            case "D":
                sum.append(sum[sum.count-1]*2)
            case "+":
                let val = sum[sum.count-1] + sum[sum.count-2]
                sum.append(val)
            default:
                sum.append(Int(letter)!)
            }
        }
        return sum.reduce(0, +)
    }

    /**
     700. Search in a Binary Search Tree
     */
    func searchBST(_ root: TreeNode?, _ val: Int) -> TreeNode? {
        func dfs(_ root: TreeNode?, _ val: Int) -> TreeNode? {
            /*
             Use rule of Binary Search Tree to get node.
             */
            guard let node = root else { return nil }
            
            if val < node.val { //check left
                return dfs(node.left, val)
            }
            else if val > node.val { //check right
                return dfs(node.right, val)
            }
            else{
                return node
            }
        }
        func bfs(_ root: TreeNode?, _ val: Int) -> TreeNode?{
            /*
             Use queue(FIFO)
             */
            var queue = [root]
            while !queue.isEmpty {
                guard let q = queue.removeFirst() else { continue }
                if q.val == val {
                    return q
                }else{
                    queue.append(q.left)
                    queue.append(q.right)
                }
            }
            return nil
        }
        
        if Bool.random(){
            return dfs(root, val)
        }else{
            return bfs(root, val)
        }
    }

    /**
     704. Binary Search
     */
    func a704search(_ nums: [Int], _ target: Int) -> Int {
        var lo = 0
        var hi = nums.count - 1
        
        while lo <= hi {
            let mid = (hi + lo) / 2
            if target > nums[mid] {
                lo = mid + 1
            }
            else if target < nums[mid]{
                hi = mid - 1
            }
            else{
                return mid
            }
        }
        return -1
    }

    /**
     709. To Lower Case
     */
    func toLowerCase(_ str: String) -> String {
        return str.lowercased()
    }
    
    
    /**
     728. Self Dividing Numbers
     */
    func selfDividingNumbers(_ left: Int, _ right: Int) -> [Int] {
        return (left...right).compactMap{ value in
            var calculated = value
            while calculated > 0{
                let divisor = calculated % 10
                guard divisor != 0 else { return nil }
                guard value % divisor == 0 else { return nil }
                calculated /= 10
            }
            return value
        }
    }
    
    /**
     766. Toeplitz Matrix
     */
    func isToeplitzMatrix_v3(_ matrix: [[Int]]) -> Bool {
        /*
         Note that loop once points which do not reach bottom or right.
         (Mean that 0..<m-1, 0..<n-1)
         Each point equals to them diagonal point.
         */
        let m = matrix.count
        let n = matrix.first?.count ?? 0
        for i in 0..<m-1 {
            for j in 0..<n-1 where matrix[i][j] != matrix[i+1][j+1] {
                return false
            }
        }
        return true
    }
    func isToeplitzMatrix_v2(_ matrix: [[Int]]) -> Bool {
        /*
         *In each diagonal all elements are the same
         *Compute index from top-left to bottom-right has the same element.
         *Edge conditions: m > 0, n >0, return true at the end of the method.
         *Reset value is -1, due to matrix[i][j] will be integers in range [0, 99]
         *Use stack and append each diagonal all points
         */
        
        let m = matrix.endIndex
        let n = matrix.first?.endIndex ?? 0
        guard m > 0, n > 0 else { return false }
        
        //Added each calculated points into stack
        var stack: [(Int, Int)] = []
        for i in 0..<m{
            stack.append((i,0))
        }
        for j in 1..<n{
            stack.append((0,j))
        }
        
        //Compute
        let reset = -1
        var cur = reset
        let bottomEdge = m-1
        let rightEdge  = n-1
        while !stack.isEmpty {
            let (i,j) = stack.removeLast()
            
            //Compare
            if cur == reset {
                cur = matrix[i][j]
            }else if cur != matrix[i][j] {
                return false
            }
            
            //Loop conditions
            if i == bottomEdge || j == rightEdge {
                cur = reset
            }else{
                let new = (i+1, j+1)
                stack.append(new)
            }
        }
        
        return true
    }
    func isToeplitzMatrix_v1(_ matrix: [[Int]]) -> Bool {
        /*
         *In each diagonal all elements are the same
         *Compute index from top-left to bottom-right has the same element.
         *Edge conditions: m > 0, n >0, return true at the end of the method.
         *Loop from bottom-left to top-right, and set while conditions.
         *Reset value is -1, due to matrix[i][j] will be integers in range [0, 99]
         */
        
        let m = matrix.endIndex
        let n = matrix.first?.endIndex ?? 0
        guard m > 0, n > 0 else { return false }
        var i = m-1
        var j = 0
        let reset = -1
        var cur = reset
        
        while !(j == n-1 && i == 0) {
            print(i, j )
            if cur == reset {
                cur = matrix[i][j]
            }else if cur != matrix[i][j] {
                return false
            }
            
            /*
             if meet the bottom(i == m-1)
             go to next J
             repeat i -= 1, j -= 1
             until j == 0 or i == 0
             and reset cur
             elsewise, i += 1, j += 1
             */
            if i == m-1 || j == n-1 {
                j += 1
                cur = reset
                while true{
                    if j == 0 || i == 0 { break }
                    i -= 1
                    j -= 1
                }
            }else{
                i += 1
                j += 1
            }
        }
        
        return true
    }

    /**
     771. Jewels and Stones
     */
    func numJewelsInStones(_ J: String, _ S: String) -> Int {
        let dict = S.reduce(into: [:]) { (counts, letter) in
            counts[letter, default:0] += 1
        }
        
        return J.compactMap{ dict[$0] }.reduce(0, +)
    }
    
    /**
     783. Minimum Distance Between BST Nodes
     */
    func minDiffInBST(_ root: TreeNode?) -> Int {
        /*
         Use inorder traversal in depth first search to get the sorted nodes(<)
         and keep previous node.
         return the minimum difference between the values of any two different nodes in the tree.
         */
        func dfs(_ root: TreeNode?, _ prev: inout TreeNode?) -> Int{
            guard let node = root else { return Int.max }
            var result = dfs(node.left, &prev)
            if let prev = prev{
                result = min(result, node.val-prev.val)
            }
            prev = node
            let rR = dfs(node.right, &prev)
            return min(result, rR)
        }
        
        var captured: TreeNode?
        return dfs(root, &captured)
    }

    /**
     812. Largest Triangle Area
     */
    struct Point: Equatable{
        let x: Int
        let y: Int
    }
    func largestTriangleArea(_ points: [[Int]]) -> Double {
        var ptx = points.map { Point(x: $0[0], y: $0[1]) }
        var value: Double = 0
        
        for i in ptx.indices.dropLast(2) {
            for j in ptx.indices.dropLast(1){
                for k in ptx.indices.dropLast(0) {
                    let (a, b, c) = (ptx[i], ptx[j], ptx[k])
                    let ac = Vector(p0: a, p1: b)
                    let ab = Vector(p0: a, p1: c)
                    value = max(value, triagnleArea(ab, ac))
                }
            }
        }
        return value
    }
    struct Vector{
        let p0: Point
        let p1: Point
        var x: Int { return p1.x - p0.x }
        var y: Int { return p1.y - p0.y }
    }
    
    func triagnleArea(_ ab: Vector, _ ac: Vector) -> Double{
        return 0.5 * Double(abs( ab.x * ac.y - ab.y * ac.x ))
    }
    
    
    /**
     844. Backspace String Compare
     */
    func backspaceCompare(_ S: String, _ T: String) -> Bool {
        return removeBackspace(S) == removeBackspace(T)
    }
    
    func removeBackspace(_ s: String) -> String{
        return s.reduce(into: "") { (r, c) in
            switch c {
            case _ where c != "#":
                r.append(c)
            case "#" where r.count > 0:
                _ = r.removeLast()
            default:
                break
            }
        }
    }

    
    /**
     874. Walking Robot Simulation
     */
    func robotSim(_ commands: [Int], _ obstacles: [[Int]]) -> Int {
        /*
         //-2: turn left 90 degrees
         //-1: turn right 90 degrees
         //1 <= x <= 9: move forward x units
         */
        typealias Point = (x: Int, y: Int)
        
        enum Direction: Int{
            case north
            case east
            case south
            case west
            
            var point: Point {
                switch self {
                case .north: return (0, 1)
                case .east : return (1, 0)
                case .south: return (0, -1)
                case .west : return (-1, 0)
                }
            }
            
            func turn(right: Bool) -> Direction{
                var next = rawValue + (right ? 1 : -1)
                next = next < 0 ? next + 4 : next
                return Direction(rawValue: next % 4)!
            }
        }
        
        var direction = Direction.north
        var point: Point = (0,0)
        var result = 0
        let obstacles = Set(obstacles)
        
        commands.forEach{ cmd in
            switch cmd {
            case -2:
                direction = direction.turn(right: false)
                
            case -1:
                direction = direction.turn(right: true)
                
            default:
                for _ in 1...cmd{
                    let next = direction.point
                    let check = [point.x + next.x, point.y + next.y]
                    guard !obstacles.contains(check) else { break }
                    point.x += next.x
                    point.y += next.y
                }
                let area = point.x * point.x + point.y * point.y
                result = max(result, area)
            }
        }
        return result
    }


    /**
     876. Middle of the Linked List
     */
    func middleNode(_ head: ListNode?) -> ListNode? {
        /*
         Think about incrasableSingle / incrasableDouble
         */
        
        var incrasableSingle = head
        var incrasableDouble = head
        
        while let _ = incrasableDouble?.next {
            incrasableSingle = incrasableSingle?.next
            incrasableDouble = incrasableDouble?.next?.next
        }
        
        return incrasableSingle
    }

    /**
     897. Increasing Order Search Tree
     */
    func increasingBST(_ root: TreeNode?) -> TreeNode? {
        /*
         Use RNL traversal in depth first search to get the sorted nodes(<)
         and keep previous node.
         */
        func dfs(_ root: TreeNode?, _ prev: inout TreeNode?) {
            guard let node = root else { return }
            dfs(node.right, &prev)
            node.right = prev
            prev?.left  = nil
            prev = node
            dfs(node.left, &prev)
        }
        
        var captured: TreeNode?
        dfs(root, &captured)
        return captured
    }

    /**
     914. X of a Kind in a Deck of Cards
     */
    func hasGroupsSizeX(_ deck: [Int]) -> Bool {
        /*
         edge condtions:
         Each group has exactly X (>=2) cards.
         All the cards in each group have the same integer.
         */
        
        //Group the same interger into dictionary
        let dict = deck.reduce(into: [:], { r, c in
            r[c, default: 0] += 1
        })
        
        //Map dictionary to the number of cards
        let numbers = dict.map{ $0.value }
        
        //Edge contdions X>=2
        if numbers.contains(1) {
            return false
        }
        
        //GCD method
        func gcd(_ a: Int, _ b: Int) -> Int {
            return (b == 0) ? a : gcd(b, a % b)
        }
        
        //Chceck all of numbers have the greatest common factor.
        for i in 0..<numbers.endIndex-1 {
            let numGCD = gcd(numbers[i], numbers[i+1])
            if numGCD == 1 {
                return false
            }else{
                continue
            }
        }
        return true
    }
    
    
    /**
     938. Range Sum of BST
     */
    func rangeSumBST(_ root: TreeNode?, _ L: Int, _ R: Int) -> Int {
        /*
         dfs, postorder
         bfs, queue
         considering bst
         */
        var sum = 0
        
        func isInRange(_ val: Int) -> Bool {
            return L <= val && val <= R
        }
        
        func dfs(_ root: TreeNode?) {
            guard let node = root else { return }
            
            if isInRange(node.val){
                sum += node.val
            }
            dfs(node.left)
            dfs(node.right)
        }
        
        func bfs(_ root: TreeNode?) {
            guard let node = root else { return }
            var queue = [node]
            while !queue.isEmpty {
                let q = queue.removeFirst()
                
                if isInRange(q.val){
                    sum += q.val
                }
                
                if let l = q.left{
                    queue.append(l)
                }
                if let r = q.right{
                    queue.append(r)
                }
            }
        }
        
        //bst + dfs
        func bst(_ root: TreeNode?, _ L: Int, _ R: Int) -> Int {
            //if root does not exist, return 0
            guard let root = root else { return 0 }
            
            //if the root is too small/big, return the range sum for its respective adjacent one..
            guard root.val >= L else { return bst(root.right, L, R) }
            guard root.val <= R else { return bst(root.left, L, R) }
            
            //find sum from leftmost/rightmost possible node to root
            let left = bst(root.left, L, R)
            let right = bst(root.right, L, R)
            
            //return the sum from left->root + root + root<-right
            return left + root.val + right
        }
        
        if Bool.random(){
            return bst(root, L, R)
        }else{
            if Bool.random(){
                dfs(root)
            }else{
                bfs(root)
            }
        }
        return sum
    }


    /**
     941. Valid Mountain Array
     */
    func validMountainArray(_ A: [Int]) -> Bool {
        /*
         A.length >= 3
         There exists some i with 0 < i < A.length - 1 such that:
         A[0] < A[1] < ... A[i-1] < A[i]
         A[i] > A[i+1] > ... > A[A.length - 1]
         */
        
        //Note that leetcode only accept one mountain.
        func mounts(_ A: [Int], count: Int) -> Bool {
            guard A.count >= 3 else { return false }
            guard A[0] < A[1] else { return false }
            var upper = 0
            var lower = 0
            var isUpper = true
            for i in 0..<A.endIndex-1 {
                if A[i] < A[i+1], isUpper {
                    upper += 1
                    isUpper.toggle()
                }
                else if A[i] > A[i+1], !isUpper{
                    lower += 1
                    isUpper.toggle()
                }
                else if A[i] == A[i+1] {
                    return false
                }
            }
            return upper == count && upper == lower
        }
        
        return mounts(A, count: 1)
    }


    /**
     942. DI String Match

     */
    func diStringMatch(_ S: String) -> [Int] {
        /*
         the count of result of array = the length of S + 1
         when get char D, append high number and subract it by 1
         elsewise
         when get char I, append low number and add it by 1
         */
        
        let s = Array(S)
        var lo = s.startIndex
        var hi = s.endIndex
        var result: [Int] = []
        
        //handle with result until the count of result equals to the length of S
        s.forEach{ c in
            if c == "I" {
                result.append(lo)
                lo += 1
            }else{
                result.append(hi)
                hi -= 1
            }
        }
        
        //Now, we can assume that lo == hi, so we can append last element into result
        result.append(lo)
        return result
    }


    /**
     965. Univalued Binary Tree
     */
    func isUnivalTree(_ root: TreeNode?) -> Bool {
        guard let node = root else { return false }
        return loop(node, value: node.val)
    }
    
    func loop(_ node: TreeNode?, value: Int) -> Bool{
        guard let node = node else { return true }
        if !(node.val == value){
            return false
        }
        if !loop(node.left, value: value) {
            return false
        }
        if !loop(node.right, value: value) {
            return false
        }
        return true
    }
    
    /**
     976. Largest Perimeter Triangle
     */
    func largestPerimeter(_ A: [Int]) -> Int {
        var sorted = A.sorted(by: >)
        for i in 0..<A.count-2 {
            guard sorted[i+0] < sorted[i+1] + sorted[i+2] else { continue }
            return sorted[i+0] + sorted[i+1] + sorted[i+2]
        }
        return 0
    }
    
    /**
     1009. Complement of Base 10 Integer
     */
    func bitwiseComplement(_ N: Int) -> Int {
        var T = 1
        while T < N {
            T |= T<<1
        }
        return N ^ T
    }

    /**
     1022. Sum of Root To Leaf Binary Numbers
     */
    func sumRootToLeaf(_ root: TreeNode?) -> Int {
        func isLeaf(_ root: TreeNode) -> Bool{
            return root.left === root.right
        }
        func bfs(_ root: TreeNode?, cur: Int) -> Int {
            guard let node = root else { return 0 }
            let sum = cur << 1 | node.val
            
            guard !isLeaf(node) else { return sum }
            let l = bfs(node.left, cur: sum)
            let r = bfs(node.right, cur: sum)
            
            return l + r
        }
        
        return bfs(root, cur: 0)
    }

    /**
     1047. Remove All Adjacent Duplicates In String
     */
    func removeDuplicates(_ S: String) -> String {
        var arr: [Character] = []
        S.forEach{ c in
            if arr.last == c {
                _ = arr.popLast()
            }else{
                arr.append(c)
            }
        }
        return String(arr)
    }

    /**
     1103. Distribute Candies to People
     */
    func distributeCandies(_ candies: Int, _ num_people: Int) -> [Int] {
        
        var result = Array(repeating: 0, count: num_people)
        var reminder = candies
        var n = 0
        
        while reminder > 0 {
            let given = n + 1
            result[n % num_people] += min(given, reminder)
            reminder -= given
            n += 1
        }
        
        return result
    }

    /**
     1108. Defanging an IP Address
     */
    func defangIPaddr(_ address: String) -> String {
        /*
         split/components .
         join [.]
         */
        return address.components(separatedBy: ".").joined(separator: "[.]")
    }

}

extension Solution{
    public class ListNode {
        public var val: Int
        public var next: ListNode?
        public init(_ val: Int) {
            self.val = val
            self.next = nil
        }
    }
    
    public class TreeNode {
        public var val: Int
        public var left: TreeNode?
        public var right: TreeNode?
        public init(_ val: Int) {
            self.val = val
            self.left = nil
            self.right = nil
        }
    }
}
extension Solution.TreeNode{
    var isLeaf: Bool { return left == nil && right == nil }
}

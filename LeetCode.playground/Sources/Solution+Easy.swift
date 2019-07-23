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
    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
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
     268. Missing Number
     */
    func missingNumber(_ nums: [Int]) -> Int {
        return nums.count * (nums.count + 1) / 2 - nums.reduce(0){ $0 + $1 }
    }
    
    
    /**
     344. Reverse String
     */
    func reverseString(_ s: String) -> String {
        return String(s.reversed())
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
     704. Binary Search
     */
    func search(_ nums: [Int], _ target: Int) -> Int {
        var lo = 0
        var hi = nums.count - 1
        
        while lo <= hi {
            var mid = (hi + lo) / 2
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
     771. Jewels and Stones
     */
    func numJewelsInStones(_ J: String, _ S: String) -> Int {
        let dict = S.reduce(into: [:]) { (counts, letter) in
            counts[letter, default:0] += 1
        }
        
        return J.compactMap{ dict[$0] }.reduce(0, +)
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
    
}

extension Solution{
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

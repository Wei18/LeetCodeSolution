//
//  Solution+Medium.swift
//
//
//  Created by Wei on 2019/07/19.
//

import Foundation

//MARK:- Medium
public extension Solution{
    
    /**
     3. Longest Substring Without Repeating Characters
     */
    func lengthOfLongestSubstring(_ s: String) -> Int {
        var res: [Character: Int] = [:]
        var r = 0
        var start = 0
        s.enumerated().forEach({ (index, letter) in
            //print(index, letter)
            if let found = res[letter], start <= found{
                start = found + 1
            }else{
                r = max(r, index - start + 1)
            }
            res[letter] = index
        })
        return r
    }
    
    
    /**
     12. Integer to Roman
     */
    func intToRoman(_ num: Int) -> String {
        enum Roman: Int{
            case I = 1
            case V = 5
            case X = 10
            case L = 50
            case C = 100
            case D = 500
            case M = 1000
            var string: String{
                switch self {
                case .I: return "I"
                case .V: return "V"
                case .X: return "X"
                case .L: return "L"
                case .C: return "C"
                case .D: return "D"
                case .M: return "M"
                }
            }
        }
        
        func replace(_ num: Int, lower: Roman, middle: Roman, upper: Roman) -> String{
            precondition(num < 10)
            var romanString = ""
            switch num {
            case 0:
                break
            case 1...3:
                romanString = (0..<num-0).reduce(romanString, { (r, _) in r + lower.string })
            case 4:
                romanString = lower.string + middle.string
            case 5...8:
                romanString = middle.string
                fallthrough
            case 6...8:
                romanString = (0..<num-5).reduce(romanString, { (r, _) in r + lower.string })
            case 9:
                romanString = lower.string + upper.string
            default:
                break
            }
            return romanString
        }
        
        let thousands =  num / 1000
        let hundreds  = (num % 1000) / 100
        let tens      = (num %  100) /  10
        let ones      = (num %   10)
        
        var r: String
        r  = replace(thousands, lower: .M, middle: .M, upper: .M)
        r += replace(hundreds, lower: .C, middle: .D, upper: .M)
        r += replace(tens, lower: .X, middle: .L, upper: .C)
        r += replace(ones, lower: .I, middle: .V, upper: .X)
        return r
    }

    /**
     102. Binary Tree Level Order Traversal
     */
    func levelOrder(_ root: TreeNode?) -> [[Int]] {
        var levelOrderArr: [[Int]] = []
        recursivelyLevelOrder(root, index: 0, result: &levelOrderArr)
        return levelOrderArr
    }
    
    func recursivelyLevelOrder(_ root: TreeNode?, index: Int, result: inout [[Int]]){
        guard let node = root else { return }
        let nextIndex = index + 1
        if index == result.count {
            result.append([node.val])
        }else{
            result[index].append(node.val)
        }
        
        recursivelyLevelOrder(node.left, index: nextIndex, result: &result)
        recursivelyLevelOrder(node.right, index: nextIndex, result: &result)
    }
    
    
    /**
     113. Path Sum II
     */
    func pathSum(_ root: TreeNode?, _ sum: Int) -> [[Int]] {
        return DFSPathSum(root, sum)
    }
    
    func DFSPathSum(_ root: TreeNode?, _ sum: Int) -> [[Int]] {
        //nested function
        func isLeaf(_ node: TreeNode) -> Bool{
            return node.left === node.right
        }
        
        guard let node = root else { return [] }
        //preorder
        if isLeaf(node), sum == node.val {
            return [[sum]]
        }else{
            let leftDFS = DFSPathSum(node.left, sum - node.val)
            let rightDFS = DFSPathSum(node.right, sum - node.val)
            return (leftDFS + rightDFS).map{ [node.val] + $0 }
        }
    }

    /**
     129. Sum Root to Leaf Numbers
     */
    
    func sumNumbers(_ root: TreeNode?) -> Int {
        return recursivelySumNumbers(root, previous: 0)
    }
    
    func recursivelySumNumbers(_ root: TreeNode?, previous: Int) -> Int{
        guard let node = root else { return 0 }
        
        func isLeaf(_ root: TreeNode) -> Bool{
            return root.left == nil && root.right == nil
        }
        
        if isLeaf(node) {
            node.val += previous
            return node.val
        }else{
            node.val = (node.val + previous) * 10
            let l = recursivelySumNumbers(node.left, previous: node.val)
            let r = recursivelySumNumbers(node.right, previous: node.val)
            return l + r
        }
    }

    /**
     260. Single Number III
     */
    func singleNumber(_ nums: [Int]) -> [Int] {
        var dict: [Int: Int] = [:]
        
        nums.forEach{ num in
            if dict[num] != nil {
                dict[num] = nil
            }else{
                dict[num] = 1
            }
        }
        return dict.map{ $0.value }
    }

    /**
     442. Find All Duplicates in an Array
     */
    func findDuplicates(_ nums: [Int]) -> [Int] {
        var nums = nums
        var res: [Int] = []
        
        nums.enumerated().forEach { (_, value) in
            let index = value - 1
            if nums[index] < 0{
                res.append(value)
            }else{
                nums[index] = -nums[index]
            }
        }
        
        return res
    }
    
    
    /**
     462. Minimum Moves to Equal Array Elements II
     */
    func minMoves2(_ nums: [Int]) -> Int {
        var n = nums.sorted()
        var r = 0
        let mi = nums.count/2
        for i in 0..<nums.count{
            let val = (mi > i) ? 1 : -1
            r+=(n[mi]-n[i])*val
        }
        return r
    }
    
    
    /**
     513. Find Bottom Left Tree Value
     */
    func findBottomLeftValue(_ root: TreeNode?) -> Int {
        let random = Bool.random()
        if random{
            //DFS:recursively
            var result: TreeNode = root!
            var maxDepth = 1
            recursivelyFindBottomLeftValue(root, depth: 1, maxDepth: &maxDepth, res: &result)
            return result.val
        }else{
            //BFS
            return BFSFindBottomLeftValue(root)
        }
    }
    
    func BFSFindBottomLeftValue(_ root: TreeNode?) -> Int{
        var q = [root!]
        var res: TreeNode!
        var index = 0
        while index < q.count {
            res = q[index]
            index += 1
            
            if let r = res.right{
                q.append(r)
            }
            if let l = res.left{
                q.append(l)
            }
        }
        return res.val
    }
    
    func recursivelyFindBottomLeftValue(_ root: TreeNode?, depth: Int, maxDepth: inout Int, res: inout TreeNode ){
        guard let node = root else { return }
        if maxDepth < depth {
            maxDepth = depth
            res = node
        }
        recursivelyFindBottomLeftValue(node.left, depth: depth + 1, maxDepth: &maxDepth, res: &res)
        recursivelyFindBottomLeftValue(node.right, depth: depth + 1, maxDepth: &maxDepth, res: &res)
    }

    
    /**
     515. Find Largest Value in Each Tree Row
     */
    func largestValues(_ root: TreeNode?) -> [Int] {
        if Bool.random(){
            var res: [Int] = []
            DFSLargestValues(root, depth: 0, result: &res)
            return res
        }else{
            return BFSLargestValues(root)
        }
    }
    
    func DFSLargestValues(_ root: TreeNode?, depth: Int, result: inout [Int]){
        guard let node = root else { return }
        //preorder
        if depth >= result.count {
            result.append(node.val)
        }else{
            result[depth] = max(result[depth], node.val)
        }
        DFSLargestValues(node.left, depth: depth + 1, result: &result)
        DFSLargestValues(node.right, depth: depth + 1, result: &result)
    }
    
    func BFSLargestValues(_ root: TreeNode?) -> [Int] {
        guard let node = root else { return [] }
        var queue = [node]
        var res: [Int] = []
        var currentFloor = 0
        
        //I wanna know the same floor of queue
        while !queue.isEmpty{
            
            //assume that its are the same floor
            for _ in queue.indices{
                let q = queue.removeFirst()
                
                if currentFloor >= res.count{
                    res.append(q.val)
                }else{
                    res[currentFloor] = max(res[currentFloor], q.val)
                }
                
                if let l = q.left{
                    queue.append(l)
                }
                if let r = q.right{
                    queue.append(r)
                }
            }
            currentFloor += 1
        }
        return res
    }

    /**
     540. Single Element in a Sorted Array
     */
    func singleNonDuplicate(_ nums: [Int]) -> Int {
        var result: Int = 0
        var preI = 0
        for var i in 0..<nums.count{
            if i == nums.count - 1{
                result = nums[i]
            }
            else if nums[i] == nums[i + 1]{
                i = i + preI
            }
            else if i > 0, nums[i] == nums[i - 1] {
                i = (preI + i) / 2
            }
            else{
                result = nums[i]
                break
            }
            preI = i
        }
        return result
    }
    
    /**
     735. Asteroid Collision
     */
    func asteroidCollision(_ asteroids: [Int]) -> [Int] {
        var r = asteroids
        var len: Int { return r.count - 1 }
        var i = 0
        
        while(i<len){
            if
                r[i] * r[i+1] < 0,
                r[i] - r[i+1] > 0 {
                switch r[i] + r[i+1] {
                case 0:
                    r.remove(at: i)
                    r.remove(at: i)
                    i-=2
                case 1...:
                    r.remove(at: i+1)
                    i-=1
                default:
                    r.remove(at: i)
                    i-=2
                }
            }
            i+=1
            if i < 0 {
                i = 0
            }
        }
        return r
    }
    
    
    /**
     763. Partition Labels
     */
    func partitionLabels(_ S: String) -> [Int] {
        var set: Set<Character> = []
        var r: [Int] = []
        var lastIndex = -1
        
        var letterCount = S.reduce(into: [:]) { counts, letter in
            counts[letter, default: 0] += 1
        }
        
        S.enumerated().forEach{ index, letter in
            letterCount[letter]! -= 1
            
            if !set.contains(letter){
                set.insert(letter)
            }
            if letterCount[letter]! > 0 {
            }
            else if set.count > 1 {
                set.remove(letter)
            }
            else{
                set.remove(letter)
                r.append( index - lastIndex )
                lastIndex = index
            }
        }
        
        return r
    }

    
    /**
     918. Maximum Sum Circular Subarray
     */
    func maxSubarraySumCircular(_ A: [Int]) -> Int {
        let sum = A.reduce(0, +)
        let aMax = kanade(A)
        let otherMax = kanade(A.map{ -$0 })
        if sum + otherMax > 0 {
            return max(aMax, sum + otherMax)
        }else{
            return aMax
        }
        
    }
    
    func kanade(_ A: [Int]) -> Int{
        var max_so_far = A.first ?? 0
        var max_ending_here = A.first ?? 0
        
        for i in 1..<A.count {
            max_ending_here = max(A[i], max_ending_here + A[i])
            max_so_far = max(max_so_far, max_ending_here)
        }
        
        return max_so_far
    }

}

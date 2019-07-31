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

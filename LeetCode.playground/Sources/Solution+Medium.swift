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
     2. Add Two Numbers
     */
    func reverseOrder_addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        let head = sumOf2List(l1, l2)
        var carry = 0
        var node = head
        while let found = node {
            found.val += carry
            carry = found.val / 10
            found.val %= 10
            if carry > 0, node?.next == nil {
                node?.next = ListNode(0)
            }
            node = node?.next
        }
        return head
    }
    
    func sumOf2List(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        guard let l1 = l1 else { return l2 }
        guard let l2 = l2 else { return l1 }
        l1.val += l2.val
        l1.next = sumOf2List(l1.next, l2.next)
        return l1
    }

    
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
    5. Longest Palindromic Substring
    
    To consider each char in s,
    should have the same value of both left char and right char,
    record the result and then loop next char.
    
    Case: even palindrome / odd palindrome
    */
    func longestPalindrome(_ s: String) -> String {
        
        func getPalindromeIndices(_ left: Int, _ right: Int) -> (Int, Int) {
            var (left, right) = (left, right)
            while left >= 0, right < chars.count, chars[left] == chars[right] {
                left -= 1
                right += 1
            }
            return (left+1, right-1)
        }
        
        var result = (0, 0)
        let chars = Array(s)
        
        for i in chars.indices {
            
            let odd = getPalindromeIndices(i, i)
            if odd.1 - odd.0 > result.1 - result.0 {
                result = odd
            }
            
            let even = getPalindromeIndices(i, i+1)
            if even.1 - even.0 > result.1 - result.0 {
                result = even
            }
            
        }
        
        return String(chars[result.0...result.1])
    }
    
    /**
     11. Container With Most Water
     */
    func maxArea(_ height: [Int]) -> Int {
        var l = height.startIndex
        var r = height.endIndex - 1
        var res = 0
        
        while l < r {
            let h = min( height[l], height[r] )
            let b = r - l
            res = max(b*h, res)
            if height[l] < height[r]{
                l += 1
            }else{
                r -= 1
            }
        }
        
        return res
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
     15. 3Sum
     */
    func threeSum(_ nums: [Int]) -> [[Int]] {
        /*
         Think about 3 ptr, i, l, r
         */
        let nums = nums.sorted(by: <)
        var res: [[Int]] = []
        
        var i = nums.startIndex
        while i < nums.endIndex {
            let iVal = nums[i]
            
            var l = i + 1
            var r = nums.endIndex - 1
            while l < r{
                let lVal = nums[l]
                let rVal = nums[r]
                
                let sum = iVal + lVal + rVal
                if sum == 0 {
                    res.append([iVal, lVal, rVal])
                }else if sum > 0{
                    r -= 1
                    continue
                }else{ //<0
                    l += 1
                    continue
                }
                
                //skip duplicates of lVal
                while l < r && nums[l] == lVal{
                    l += 1
                }
                //skip duplicates of rVal
                while l < r && nums[r] == rVal{
                    r -= 1
                }
            }
            
            //skip duplicates of iVal
            while i < nums.endIndex && nums[i] == iVal{
                i += 1
            }
        }
        
        return res
    }

    
    /**
     17. Letter Combinations of a Phone Number
     */
    func letterCombinations(_ digits: String) -> [String] {
        guard !digits.isEmpty else { return [] }
        var mapping = [Character: [String]]()
        mapping["2"] = ["a", "b", "c"]
        mapping["3"] = ["d", "e", "f"]
        mapping["4"] = ["g", "h", "i"]
        mapping["5"] = ["j", "k", "l"]
        mapping["6"] = ["m", "n", "o"]
        mapping["7"] = ["p", "q", "r", "s"]
        mapping["8"] = ["t", "u", "v"]
        mapping["9"] = ["w", "x", "y", "z"]
        
        func mix(_ a: [String], _ b: [String]) -> [String]{
            guard !a.isEmpty else { return b }
            var res: [String] = []
            for i in a.indices{
                res += b.map{ a[i] + $0 }
            }
            return res
        }
        
        let mappingRes = digits.compactMap{ mapping[$0] }
        var res = [String]()
        for i in 0..<mappingRes.endIndex{
            res = mix(res, mappingRes[i])
        }
        
        return res
    }

    
    /**
     19. Remove Nth Node From End of List
     */
    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        if Bool.random(){
            let current = DFSRemoveNthFromEnd(head, n)
            return (current == n) ? head?.next : head
        }else{
            return NorRemoveNthFromEnd(head, n)
        }
    }
    
    func DFSRemoveNthFromEnd(_ root: ListNode?, _ n: Int) -> Int{
        guard let node = root?.next else { return 1 }
        let current = DFSRemoveNthFromEnd(node, n)
        if current == n{
            root?.next = root?.next?.next
        }
        return current + 1
    }
    
    func NorRemoveNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        var stack: [ListNode] = []
        
        var node = head
        while let found = node {
            stack.append(found)
            node = node?.next
        }
        
        //or stack.remove(at: stack.count - n)
        var removed: ListNode?
        for _ in 0..<n{
            removed = stack.popLast()
        }
        
        let currentList = stack.popLast()
        currentList?.next = removed?.next
        
        if currentList == nil {
            return removed?.next
        }else{
            return head
        }
    }

    /**
     24. Swap Nodes in Pairs
     */
    func swapPairs(_ head: ListNode?) -> ListNode? {
        /*
         1. Set edge condition of recursive method.
         2. Recursively use method to reach last pair.
         3. Swap 2 nodes even and odd, bind rest of nodes and the odd node and then return the node even.
         */
        guard let odd = head, let even = odd.next else { return head }
        let rest = swapPairs(even.next)
        even.next = odd
        odd.next = rest
        return even
    }

    /**
     33. Search in Rotated Sorted Array
     */
    func search(_ nums: [Int], _ target: Int) -> Int {
        //2 ptr: l, r
        var l = nums.startIndex
        var r = nums.endIndex-1
        while l <= r {
            if nums[l] == target {
                return l
            }
            else if nums[r] == target{
                return r
            }
            else{
                r -= 1
                l += 1
            }
        }
        return -1
    }

    
    /**
     54. Spiral Matrix
     */
    func spiralOrder(_ matrix: [[Int]]) -> [Int] {
        /*
         Think about 4 directions of up, right, down, left, and then loop it.
         Check matrix is visited
         */
        let m = matrix.count
        let n = matrix.first?.count ?? 0
        guard m > 0, n > 0 else { return [] }
        var visited = Array(repeating: Array(repeating: false, count: n), count: m)
        var result: [Int] = []
        
        func isAllVisited(_ x: Int, _ y: Int) -> Bool {
            //top
            if y > 0, !visited[x][y-1] {
                return false
            }
            //right
            if x+1 < m, !visited[x+1][y] {
                return false
            }
            //bottom
            if y+1 < n, !visited[x][y+1] {
                return false
            }
            //left
            if x > 0, !visited[x-1][y] {
                return false
            }
            //self
            if !visited[x][y] {
                return false
            }else{
                return true
            }
        }
        
        func record(_ x: Int, _ y: Int){
            result.append(matrix[x][y])
            visited[x][y] = true
        }
        
        enum Way: Int{
            case right
            case down
            case left
            case up
        }
        
        var i = 0
        var j = 0
        var way = Way.right
        
        while !isAllVisited(i, j) {
            //record result
            record(i, j)
            
            //next point
            switch way {
            case .right where j+1 < n && !visited[i][j+1]:
                j += 1
            case .right where i+1 < m:
                i += 1
                way = .down
                
            case .down where i+1 < m && !visited[i+1][j]:
                i += 1
            case .down where j > 0:
                j -= 1
                way = .left
                
            case .left where j > 0 && !visited[i][j-1]:
                j -= 1
            case .left where i > 0:
                i -= 1
                way = .up
                
            case .up where i > 0 && !visited[i-1][j]:
                i -= 1
            case .up where j+1 < n:
                j += 1
                way = .right
                
            default:
                break
            }
        }
        
        return result
    }

    
    /**
     61. Rotate List
     */
    func rotateRight(_ head: ListNode?, _ k: Int) -> ListNode? {
        func length(_ head: ListNode?) -> Int{
            var count = 0
            var node = head
            while let _ = node{
                count += 1
                node = node?.next
            }
            return count
        }
        
        let count = length(head)
        let offset = k % count
        guard offset > 0 else { return head }
        var slow = head
        var fast = head
        
        //okay, I got the offset node.
        //k: 5, offset: 0, fast: 1, slow: 1
        //k: 1, offset: 1, fast: 2, slow: 1
        //k: 2, offset: 2, fast: 3, slow: 1
        //k: 3, offset: 3, fast: 4, slow: 1
        //k: 4, offset: 4, fast: 5, slow: 1
        for _ in 0..<offset{
            fast = fast?.next
        }
        
        //now I need reach last node.
        //k: 5, offset: 0, fast: 5, slow: 5, rotatedHead = slow.nex
        //k: 1, offset: 1, fast: 5, slow: 4, rotatedHead = slow.nex
        //k: 2, offset: 2, fast: 5, slow: 3, rotatedHead = slow.nex
        //k: 3, offset: 3, fast: 5, slow: 2, rotatedHead = slow.nex
        //k: 4, offset: 4, fast: 5, slow: 1, rotatedHead = slow.nex
        while fast?.next != nil {
            slow = slow?.next
            fast = fast?.next
        }
        
        //loop list
        fast?.next = head
        //get rotatedHead
        let new_head = slow?.next
        //break loop
        slow?.next = nil
        
        return new_head
        
        /*
         Input: 1->2->3->4->5->NULL, k = 1
         Explanation:
         rotate 1 steps to the right: 5->1->2->3->4->NULL
         rotate 2 steps to the right: 4->5->1->2->3->NULL
         */
    }

    /**
     74. Search a 2D Matrix
     */
    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
        /*
         Use binary search, and split mid into i,j
         */
        
        let m = matrix.count
        guard m > 0, let n = matrix.first?.count, n > 0 else { return false }
        var l = 0
        var r = m * n - 1
        
        func coordinate(_ index: Int) -> Int {
            let i = index / n
            let j = index % n
            return matrix[i][j]
        }
        
        while l <= r{
            let mid = (l + r) / 2
            let value = coordinate(mid)
            if value == target {
                return true
            }else if value > target {
                r = mid - 1
            }else {
                l = mid + 1
            }
        }
        
        return false
    }

    /**
     94. Binary Tree Inorder Traversal
     */
    func inorderTraversal(_ root: TreeNode?) -> [Int] {
        if Bool.random(){
            return inorderTraversal_DFS(root)
        }
        else{
            return inorderTraversal_BFS(root)
        }
    }
    func inorderTraversal_BFS(_ root: TreeNode?) -> [Int] {
        //BFS, Iterative solution, inorder?
        guard let node = root else { return [] }
        var result: [Int] = []
        var stack: [TreeNode] = []
        
        func push(_ n: TreeNode?) {
            guard let node = n else { return }
            stack.append(node)
            push(node.left)
        }
        
        push(node)
        while !stack.isEmpty {
            let pop = stack.removeLast()
            result.append(pop.val)
            push(pop.right)
        }
        
        return result
    }
    func inorderTraversal_DFS(_ root: TreeNode?) -> [Int] {
        //DFS, Recursive solution, inorder
        guard let node = root else { return [] }
        let lA = inorderTraversal_DFS(node.left)
        let cur = [node.val]
        let rA = inorderTraversal_DFS(node.right)
        return lA + cur + rA
    }

    
    /**
     130. Surrounded Regions
     */
    func solve(_ board: inout [[Character]]) {
        /*
         let's start 3 loops
         round 1, from left-top, change any 'O' on the border and it is connected to an 'O' on the border to '#'
         round 2, from right-bottom, change any 'O' on the border and it is connected to an 'O' on the border to '#'
         round 3, change any '#' on the border to 'O' and others to 'X'
         */
        guard !board.isEmpty, !board[0].isEmpty else { return }
        
        let Tag: Character = "#"
        let O: Character = "O"
        let X: Character = "X"
        let m = board.count
        let n = board[0].count
        
        //round 1
        for i in 0..<m{
            for j in 0..<n where board[i][j] == O {
                if i == 0 || j == 0 {
                    board[i][j] = Tag
                }
                else if board[i-1][j] == Tag {
                    board[i][j] = Tag
                }
                else if board[i][j-1] == Tag {
                    board[i][j] = Tag
                }
                /*
                 //round 2
                 else if i == m-1 || j == n-1 {
                 board[i][j] = "#"
                 }
                 else if board[i+1][j] == "#" {
                 board[i][j] = "#"
                 }
                 else if board[i][j+1] == "#" {
                 board[i][j] = "#"
                 }
                 */
            }
        }
        
        //round 2
        for i in stride(from: m-1, to: -1, by: -1){
            for j in stride(from: n-1, to: -1, by: -1) where board[i][j] == O {
                if i == m-1 || j == n-1 {
                    board[i][j] = Tag
                }
                else if board[i+1][j] == Tag {
                    board[i][j] = Tag
                }
                else if board[i][j+1] == Tag {
                    board[i][j] = Tag
                }
            }
        }
        
        //round 3
        for i in 0..<m{
            for j in 0..<n {
                if board[i][j] == Tag {
                    board[i][j] = O
                }else if board[i][j] == O {
                    board[i][j] = X
                }
            }
        }
    }


    /**
     98. Validate Binary Search Tree
     */
    func isValidBST(_ root: TreeNode?) -> Bool {
        /*
         Using Depth Search First, PreOrder, Recursive
         Definition edge: return true
         Compute r.val > n.val > l.vale
         */
        
        func isValidBST_rec(_ root: TreeNode?, _ minVal: Int?, _ maxVal: Int?) -> Bool {
            guard let node = root else { return true }
            
            if let minVal = minVal, node.val <= minVal {
                return false
            }
            else if let maxVal = maxVal, node.val >= maxVal {
                return false
            }
            else{
                return isValidBST_rec(node.left, minVal, node.val)
                    && isValidBST_rec(node.right, node.val, maxVal)
            }
        }
        return isValidBST_rec(root, nil, nil)
    }

    /**
     102. Binary Tree Level Order Traversal
     https://leetcode.com/problems/binary-tree-level-order-traversal/
     */
    func levelOrder(_ root: TreeNode?) -> [[Int]] {
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
        func bfs() -> [[Int]] {
            guard let node = root else { return [] }
            var queue = [node]
            var result: [[Int]] = []
            while !queue.isEmpty{
                result.append([])
                for _ in queue.indices{
                    let q = queue.removeFirst()
                    result[result.count-1].append(q.val)
                    if let l = q.left{
                        queue.append(l)
                    }
                    if let r = q.right{
                        queue.append(r)
                    }
                }
            }
            return result
        }
        if Bool.random(){
        var levelOrderArr: [[Int]] = []
            recursivelyLevelOrder(root, index: 0, result: &levelOrderArr)
            return levelOrderArr
        }else{
            return bfs()
        }
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
     148. Sort List
     */
    class MergeSortListNode{
        func sortList(_ head: ListNode?) -> ListNode? {
            /*
             get last node.
             use merge sort
             divide list to 2 list.
             merge its
             */
            var last = head
            while let node = last?.next {
                last = node
            }
            return mergeSort(head, last)
        }
        
        private func mergeSort(_ lo: ListNode?, _ hi: ListNode?) -> ListNode?{
            guard lo !== hi else { return lo }
            let (before, middle) = divide(lo, hi)
            let l: ListNode? = mergeSort(lo, before)
            let r: ListNode? = mergeSort(middle, hi)
            return merge(l, r)
        }
        
        private func divide(_ lo: ListNode?, _ hi: ListNode?) -> (ListNode?, ListNode?){
            var single = lo
            var double = lo?.next
            while double !== hi, let fF = double?.next?.next {
                single = single?.next //1 2 3 4
                double = fF           //2 4 6 8
            }
            let before = single
            let middle = single?.next
            before?.next = nil //In order to split 2 lists.
            return (before, middle)
        }
        
        
        private func merge(_ l: ListNode?, _ r: ListNode?) -> ListNode? {
            guard let l = l else { return r }
            guard let r = r else { return l }
            if l.val < r.val {
                l.next = merge(l.next, r)
                return l
            }else{
                r.next = merge(r.next, l)
                return r
            }
        }

    }

    /**
     162. Find Peak Element
     */
    func findPeakElement(_ nums: [Int]) -> Int {
        /*
         Use Iterative Binary Search from left, right
         return index of satisfied requirement
         otherwise index (l, r) will approach each other.
         */
        var l = nums.startIndex
        var r = nums.endIndex-1
        
        while l < r {
            let m = (l + r) / 2
            if nums[m] < nums[m+1] {
                l = m + 1
            }else{
                r = m
            }
        }
        
        return l
    }

    /**
     228. Summary Ranges
     */
    func summaryRanges(_ nums: [Int]) -> [String] {
        /*
         Problem:
         Given a sorted integer array without duplicates, return the summary of its ranges.
         
         Note:
         0. Edge condition
         1. Loop once
         2. Result
         3. Append string of combined start and end
         4. Loop conditions:
         the current number-1 != end, compute result
         */
        
        guard !nums.isEmpty else { return [] }
        guard nums.count > 1 else { return ["\(nums[0])"] }
        
        var result: [String] = []
        var start = nums.first
        var end = nums.first
        
        func update(){
            let text = (start == end) ? "\(start!)" : "\(start!)->\(end!)"
            result.append(text)
        }
        
        for i in 1..<nums.endIndex {
            if nums[i] - 1 != end {
                update()
                start = nums[i]
            }
            end = nums[i]
        }
        update()
        return result
    }

    
    /**
     238. Product of Array Except Self
     */
    func productExceptSelf(_ nums: [Int]) -> [Int] {
        /*
         Use 2 pointer l, r and product excpt self
         example:
         leftCurrent = 1
         nums   = [    2,   3,   4,     5]
         result = [    1,   2, 2*3, 2*3*4]
         
         rightCurrent = 1
         nums   = [    2,   3,   4,     5]
         result = [5*4*3, 5*4,   5,     1]
         
         return the result is calculated that we product left result and right result.
         */
        
        var dp = Array(repeating: 0, count: nums.count)
        
        var cur = 1
        for i in nums.indices{
            dp[i] = cur
            cur *= nums[i]
        }
        
        cur = 1
        for j in stride(from: nums.count-1, to: -1, by: -1){
            dp[j] *= cur
            cur *= nums[j]
        }
        
        return dp
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
     322. Coin Change
     */
    func coinChange(_ coins: [Int], _ amount: Int) -> Int {
        //edge
        guard amount > 0, !coins.isEmpty else { return 0 }
        //initial dp
        var dp = [Int](repeatElement(Int.max, count: amount + 1))
        dp[0] = 0
        //dp[2] == 1(coin 2) or 2(2coins 1)
        //compute conditions
        /*
         loop amount from 1 to amount
         and then loop coins to calculate minimum coins to reach amount.
         */
        for i in 1...amount {
            for j in 0..<coins.count {
                if coins[j] > i {
                    //no combination coins to reach current amount.
                    continue
                }
                else if dp[i - coins[j]] == Int.max{
                    //the index of (i - coins[j]) should have been calculated,
                    //if value == Int.max, we dont need it.
                    continue
                }
                else{
                    let currentCount = dp[i - coins[j]] + 1
                    let previousCount = dp[i]
                    dp[i] = min(currentCount, previousCount)
                }
            }
        }
        
        return dp[amount] == Int.max ? -1 : dp[amount]
    }

    
    /**
     328. Odd Even Linked List
     */
    func oddEvenList(_ head: ListNode?) -> ListNode? {
        /*
         loop
         get first odd and first even nodes
         group all odd and even nodes
         at least, added first even node at the end of odd node.
         */
        var oddList: ListNode? = head
        var evenList: ListNode? = head?.next
        let firstOdd  = oddList
        let firstEven = evenList
        while let odd = oddList?.next?.next {
            oddList?.next = odd
            oddList = odd
            
            let even = evenList?.next?.next
            evenList?.next = even
            evenList = even
        }
        oddList?.next = firstEven
        return firstOdd
    }

    
    /**
     347. Top K Frequent Elements
     */
    func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
        let r = nums
            .reduce(into: [:], { (r, i) in
                r[i, default: 0] += 1
            })
            .sorted(by: { $0.value > $1.value })
            .map{ $0.key }
            .prefix(k)
        return Array(r)
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
     445. Add Two Numbers II
     */
    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        if l1 == nil{
            return l2
        }else if l2 == nil{
            return l1
        }
        
        var stack1 = [l1!]
        var stack2 = [l2!]
        
        var tmp: ListNode?
        tmp = l1
        while let next = tmp?.next{
            stack1.append(next)
            tmp = next
        }
        
        tmp = l2
        while let next = tmp?.next{
            stack2.append(next)
            tmp = next
        }
        
        var carry = 0
        var last: ListNode?
        while !stack1.isEmpty || !stack2.isEmpty {
            let s1 = stack1.popLast()
            let s2 = stack2.popLast()
            var sum = (s1?.val ?? 0) + (s2?.val ?? 0)
            if carry != 0 {
                sum += carry
                carry = 0
            }
            if sum >= 10{
                carry = sum / 10
                sum -= 10
            }
            s1?.val = sum
            s2?.val = sum
            last = s1 ?? s2
        }
        
        if carry > 0 {
            let new = ListNode(carry)
            new.next = last
            return new
        }else{
            return last
        }
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
     470. Implement Rand10() Using Rand7()
     */
    func rand10() -> Int {
        // /*
        func rand7() -> Int {
            return Int.random(in: 0...7)
        }
        // */
        /*
         1. get the larger ragne
         and the chance for generating a random integer
         should be same.
         2. get the largest number that number % 10 == 0
         3. add 1 to match result from [0...9] to [1...10]
         */
        func rand49() -> Int{
            //[1...7] + [0...6]*7 = [1...49]
            return rand7() + (rand7()-1) * 7
        }
        
        var x: Int
        repeat {
            x = rand49()
        } while x > 40
        
        return x % 10 + 1
    }

    /**
     498. Diagonal Traverse
     */
    func findDiagonalOrder(_ matrix: [[Int]]) -> [Int] {
        /*
         Input:
         [
         [ 1, 2, 3 ],
         [ 4, 5, 6 ],
         [ 7, 8, 9 ]
         ]
         
         Output:  [1,2,4,7,5,3,6,8,9]
         
         Solution:
         Set the enum of way, top-right, bottom-left.
         Set these two edge to trun way as below
         if way == .topRight and ( i == 0 || j+1 == N )
         if way == .bottomLeft and ( i+1 == M || j == 0 )
         Set the stop condition that mean i+1 == M, j+1 == N
         Compute 2 moveing pointers
         
         Forgot:
         Additional conditions of edge:
         i == 0 && j+1 != n && way == .topRight:
         j == 0 && i+1 != m && way == .bottomLeft:
         
         Add:
         if sort cases, can remove additional conditions, become:
         case (i,j) where j+1 == n && way == .topRight:
         case (i,j) where i == 0 && way == .topRight:
         case (i,j) where i+1 == m && way == .bottomLeft:
         case (i,j) where j == 0 && way == .bottomLeft:
         */
        enum Way{
            case topRight
            case bottomLeft
            mutating func toggle() {
                if self == .bottomLeft {
                    self = .topRight
                }else{
                    self = .bottomLeft
                }
            }
        }
        let m = matrix.count
        let n = matrix.first?.count ?? 0
        var result: [Int] = []
        var way = Way.topRight
        guard m > 0, n > 0 else { return result }
        
        var i = 0
        var j = 0
        while i<m && j<n {
            result.append(matrix[i][j])
            switch (i,j) {
            case (i,j) where j+1 == n && way == .topRight:
                i += 1
                way.toggle()
            case (i,j) where i == 0 && way == .topRight:
                j += 1
                way.toggle()
            case (i,j) where i+1 == m && way == .bottomLeft:
                j += 1
                way.toggle()
            case (i,j) where j == 0 && way == .bottomLeft:
                i += 1
                way.toggle()
            default:
                if way == .bottomLeft{
                    i += 1
                    j -= 1
                }else{
                    i -= 1
                    j += 1
                }
            }
        }
        return result
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
     522. Longest Uncommon Subsequence II
     */
    func findLUSlength(_ strs: [String]) -> Int {
        /*
         define edge:
         return -1 if result is not found else return result.
         and found the subsequence of strings like the maximum common factor
         */
        
        func isSubsequence(_ subsequence: String, _ text: String) -> Bool {
            guard subsequence.count <= text.count else { return false }
            var i = subsequence.startIndex
            var j = text.startIndex
            while i < subsequence.endIndex, j < text.endIndex {
                if subsequence[i] == text[j] {
                    i = subsequence.index(after: i)
                }
                j = text.index(after: j)
            }
            return i == subsequence.endIndex
        }
        
        var strs = strs.sorted(by: { $0.count > $1.count })
        for i in strs.indices{
            let substring = strs[i]
            var isResult = true
            
            for j in strs.indices where i != j && isSubsequence(substring, strs[j]) {
                isResult = false
                break
            }
            if isResult{
                return substring.count
            }
        }
        return -1
    }


    /**
     540. Single Element in a Sorted Array
     */
    func singleNonDuplicate(_ nums: [Int]) -> Int {
        func v190905() -> Int{
            /* Binary Search */
            var l = nums.startIndex
            var r = nums.endIndex-1
            while l<=r{
                let m = (l+r) >> 1
                if m == nums.startIndex ||
                    m == nums.endIndex-1 ||
                    (nums[m-1] != nums[m] && nums[m] != nums[m+1]) {
                    return nums[m]
                }
                let match = (m%2 == 0) ? nums[m+1] : nums[m-1]
                if nums[m] == match{
                    l = m+1
                }else{
                    r = m-1
                }
            }
            return nums.last ?? 0
        }
        func v0() -> Int{
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
        return v190905()
    }
    
    
    /**
     560. Subarray Sum Equals K
     */
    func subarraySum(_ nums: [Int], _ k: Int) -> Int {
        /*
         if K equalt to sum[i, j]
         that means  sum[i, j] = sum[0, j] - sum[0, i - 1],
         rangeSum  = totalSum  - preSum
         we have rangeSum, K and totalSum sum, we can find preSum.
         so we record each of sum in loop.
         if one of recorded sum equals to preSum, resultcount ++
         */
        var result = 0
        var sum = 0
        var dict: [Int: Int] = [:]
        for i in nums.indices{
            dict[sum, default: 0] += 1
            sum += nums[i]
            if let existCount = dict[sum-k] {
                result += existCount
            }
        }
        return result
    }

    
    /**
     623. Add One Row to Tree
     */
    func DFS_addOneRow(_ root: TreeNode?, _ v: Int, _ d: Int){
        guard let node = root else { return }
        if d == 1 {
            let newLeft = TreeNode(v)
            newLeft.left = root?.left
            node.left = newLeft
            
            let newRight = TreeNode(v)
            newRight.right = root?.right
            node.right = newRight
        }
        else{
            DFS_addOneRow(node.left, v, d-1)
            DFS_addOneRow(node.right, v, d-1)
        }
    }
    func BFS_addOneRow(_ root: TreeNode?, _ v: Int, _ d: Int) -> TreeNode? {
        guard let found = root else { return nil }
        var queue = [found]
        var depth = 0
        
        while !queue.isEmpty {
            let size = queue.endIndex
            depth += 1
            
            for _ in 0..<size {
                let node = queue.removeFirst()
                
                if let left = node.left {
                    queue.append(left)
                }
                if let right = node.right {
                    queue.append(right)
                }
                
                guard depth == d - 1 else { continue }
                
                let newLeft = TreeNode(v)
                newLeft.left = node.left
                node.left = newLeft
                
                let newRight = TreeNode(v)
                newRight.right = node.right
                node.right = newRight
            }
        }
        return root
    }
    func addOneRow(_ root: TreeNode?, _ v: Int, _ d: Int) -> TreeNode? {
        /*
         Use recursive + inOrder,
         define edge conditions.
         if current depth == 1, assign new node with val is v to left and right.
         */
        if d == 1 {
            let node = TreeNode(v)
            node.left = root
            return node
        }else{
            if Bool.random() {
                DFS_addOneRow(root, v, d-1)
                return root
            }else{
                return BFS_addOneRow(root, v, d)
            }
        }
    }

    
    /**
     646. Maximum Length of Pair Chain
     */
    func findLongestChain(_ pairs: [[Int]]) -> Int {
        /*
         first step, sorted pairs
         and then increasing count if satisfied conditions, record the max.
         that's easy way.
         
         so we use dynamic programming instead of it.
         */
        
        if Bool.random(){
            let pairs = pairs.sorted(by: { $0[1] < $1[1] })
            var endingHere = Int.min
            var res = 0
            
            for pair in pairs where pair[0] > endingHere {
                res += 1
                endingHere = pair[1]
            }
            return res
        }
        else{
            let pairs = pairs.sorted(by: { $0[1] < $1[1] })
            var dp = Array(repeating: 1, count: pairs.count)
            var res = Int.min
            
            for i in pairs.indices{
                for j in 0..<i where pairs[j][1] < pairs[i][0] {
                    dp[i] = max(dp[i], dp[j]+1)
                }
                res = max(res, dp[i])
            }
            return res
        }
    }

    /**
     695. Max Area of Island
     */
    func maxAreaOfIsland(_ grid: [[Int]]) -> Int {
        /*
         Loop all of grid which is island.
         If meet island we calculate size of island by bfs and become it to water.
         Record the maximum size and return it.
         */
        
        var maxArea = 0
        var grid = grid
        let m = grid.count
        let n = grid.first?.count ?? 0
        
        func infection(y: Int, x: Int, m: Int, n: Int) -> Int {
            //horizontal conditions, vertical contitions
            guard x >= 0, x < n else { return 0 }
            guard y >= 0, y < m else { return 0 }
            
            //keep size of island and become it to water
            var size = grid[y][x]
            guard size == 1 else { return 0 }
            grid[y][x] = 0
            size += infection(y: y+1, x: x, m: m, n: n)
            size += infection(y: y-1, x: x, m: m, n: n)
            size += infection(y: y, x: x+1, m: m, n: n)
            size += infection(y: y, x: x-1, m: m, n: n)
            return size
        }
        
        for i in grid.indices{
            for j in grid[i].indices where grid[i][j] == 1 {
                let area = infection(y: i, x: j, m: m, n: n)
                maxArea = max(maxArea, area)
            }
        }
        
        return maxArea
    }

    /**
     725. Split Linked List in Parts
     */
    func splitListToParts(_ root: ListNode?, _ k: Int) -> [ListNode?] {
        /*
         get len of root
         get division of k / len
         handle with it
         */
        
        func len(_ root: ListNode?) -> Int{
            var count = 0
            var node = root
            while let _ = node {
                count += 1
                node = node?.next
            }
            return count
        }
        
        func split(_ root: ListNode?, count: Int) -> [ListNode?] {
            var res: [ListNode?] = [root, nil]
            var node = root
            var prev = node
            
            for _ in 0..<count where node != nil {
                prev = node
                node = node?.next
            }
            prev?.next = nil
            res[1] = node
            return res
        }
        
        let count = len(root)
        var result: [ListNode?] = []
        
        var disivions = Array(repeating: 0, count: k)
        for i in 0..<count{
            disivions[i%k] += 1
        }
        
        var prev: ListNode? = root
        for num in disivions{
            let splited = split(prev, count: num)
            result.append(splited[0])
            prev = splited[1]
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
     739. Daily Temperatures
     */
    func dailyTemperatures(_ T: [Int]) -> [Int] {
        /*
         record res of array of int and count of array is equal to array T
         using 2 ptrs: i, j
         loop i
         loop j < T.endIndex
         recode index of j - i of first value of T[i] < T[j]
         added: skip while curValue == nextValue
         added: hanlde index -1 for sameValue
         */
        
        var res = Array(repeating: 0, count: T.count)
        var i = T.startIndex
        while i < T.endIndex {
            var j = i + 1
            let cur = T[i]
            while j < T.endIndex {
                let future = T[j]
                if cur < future {
                    res[i] = j - i
                    break
                }
                j += 1
            }
            
            while i+1 < T.endIndex, cur == T[i+1]{
                res[i+1] = max(0, res[i] - 1)
                i += 1
            }
            i += 1
        }
        return res
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
     814. Binary Tree Pruning
     */
    func pruneTree(_ root: TreeNode?) -> TreeNode? {
        func isLeaf(_ root: TreeNode) -> Bool{
            return root.left === root.right
        }
        func isPruning(_ root: TreeNode) -> Bool{
            return root.val == 0 && isLeaf(root)
        }
        /*
         Sould remove node which is leaf and value ss equal to 0.
         Use recursive and one of postorder, inorder, preorder.
         */
        
        //edge
        guard let node = root else { return nil }
        
        //recursive, preorder
        if let left = pruneTree(node.left), isPruning(left) {
            node.left = nil
        }
        if let right = pruneTree(node.right), isPruning(right) {
            node.right = nil
        }
        
        return node
    }

    
    /**
     870. Advantage Shuffle
     */
    func advantageCount(_ A: [Int], _ B: [Int]) -> [Int] {
        let PICKED = -1 //Flag
        var A = A.sorted()
        var result: [Int] = []
        
        var iA = A.startIndex
        for b in B {
            var isMatch = false
            
            //Move iterator
            if b < (result.last ?? PICKED) {
                iA = A.startIndex
            }
            
            //Append one element that A[i] > b into array result.
            for i in iA..<A.endIndex where A[i] > b {
                result.append(A[i])
                A[i] = PICKED
                isMatch = true
                iA = i
                break
            }
            
            //Otherwise, append first element that A[i] != PICKED into array result.
            for i in A.indices where !isMatch && A[i] != PICKED {
                result.append(A[i])
                A[i] = PICKED
                break
            }
        }
        
        return result
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
    
    
    /**
     939. Minimum Area Rectangle
     */
    func minAreaRect(_ points: [[Int]]) -> Int {
        /*
         Base on a point, p1 and find the other point, p2
         that x2 > x1, y2 > y1, and point (x1, y2) and (x2, y1) esxit,
         compute min area.
         */
        
        let ptsSet = Set<[Int]>(points)
        
        let pts = points
            .sorted(by: { $0[1] < $1[1] })
            .sorted(by: { $0[0] < $1[0] })
        
        var minArea = Int.max
        
        for i in 0..<pts.endIndex-1{
            let (x1, y1) = (pts[i][0], pts[i][1])
            for j in i+1..<pts.endIndex{
                let (x2, y2) = (pts[j][0], pts[j][1])
                guard x2 > x1, y2 > y1 else { continue }
                guard ptsSet.contains([x1, y2]) else { continue }
                guard ptsSet.contains([x2, y1]) else { continue }
                let area = (x2 - x1) * (y2 - y1)
                minArea = min(minArea, area)
            }
        }
        
        return (minArea == Int.max) ? 0 : minArea
    }
    
    /**
     954. Array of Doubled Pairs
     */
    func test_canReorderDoubled(){
        assert(canReorderDoubled([3,1,3,6]) == false)
        assert(canReorderDoubled([2,1,2,6]) == false)
        assert(canReorderDoubled([4,-2,2,-4]) == true)
        assert(canReorderDoubled([0,0]) == true)
    }
    func canReorderDoubled(_ A: [Int]) -> Bool {
        func compute(_ a: [Int]) -> Bool {
            let a = a.sorted()
            var dp: [Int: Int] = a.reduce(into: [:]) { r, i in
                r[i, default: 0] += 1
            }
            
            for i in a.indices where dp[a[i]] != Int.min {
                let v = a[i]
                if dp[v*2] == nil {
                    return false
                }
                if dp[v*2] == dp[v] {
                    dp[v*2] = Int.min
                }else{
                    dp[v*2]! -= dp[v]!
                }
                dp[v] = Int.min
            }
            //All of dp values equal to Int.min after for-loop = ture
            return dp.values.first{ $0 != Int.min } == nil
        }
        
        let negArr = A.filter { $0 < 0 }.map { -$0 }
        let posArr = A.filter { $0 >= 0 }
        return compute(negArr) && compute(posArr)
    }
    
    
    /**
     958. Check Completeness of a Binary Tree
     */
    func isCompleteTree(_ root: TreeNode?) -> Bool {
        /*
         Use Breath First Search
         Mark flag "last" true if found the node is nil
         if still has nxet node return false
         return false
         */
        
        var queue = [root]
        var last = (root == nil)
        
        while !queue.isEmpty{
            let q = queue.removeFirst()
            
            if q == nil {
                last = (q == nil)
            }else if last {
                return false
            }else{
                queue.append(q?.left)
                queue.append(q?.right)
            }
        }
        
        return true
    }

    
    /**
     1014. Best Sightseeing Pair
     */
    func maxScoreSightseeingPair(_ A: [Int]) -> Int {
        /*
         dp[i][j] = A[i] + A[j] + i - j
         
         Note that Similar Kanade Algorithm, get 2 maximum values
         */
        var max_ending_here = Int.min
        var max_so_far = Int.min
        for k in 1..<A.endIndex {
            let i = A[k-1]+k-1
            let j = A[k]-k
            max_ending_here = max(max_ending_here, i)
            max_so_far = max(max_so_far, max_ending_here+j)
        }
        return max_so_far
    }

    
    /**
     1143. Longest Common Subsequence
     */
    func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
        //time complexity O(m*n)
        func omn() -> Int{
            guard text1.count > 0, text2.count > 0 else { return 0 }
            var dp = Array(repeating: Array(repeating: 0, count: text2.count+1), count: text1.count+1)
            var text1 = Array(text1)
            var text2 = Array(text2)
            
            for i in 1...text1.count {
                for j in 1...text2.count {
                    if text1[i-1] == text2[j-1] {
                        dp[i][j] = 1 + dp[i-1][j-1]
                    }else{
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    }
                }
            }
            return dp[text1.count][text2.count]
        }
        
        //time complexity O(n)
    //     func on() -> Int{
    //         guard text1.count > 0, text2.count > 0 else { return 0 }
    //         var dp = Array(repeating: 0, count: text2.count+1)
    //         var text1 = Array(text1)
    //         var text2 = Array(text2)
            
    //         for i in 1...text1.count {
    //             for j in 1...text2.count {
    //                 if text1[i-1] == text2[j-1] {
    //                     dp[j] = 1 + dp[j-1]
    //                 }else{
    //                     dp[j] = max(dp[j], dp[j-1])
    //                 }
    //             }
    //         }
    //         return dp[text2.count]
    //     }
        
        return omn()
    }



}

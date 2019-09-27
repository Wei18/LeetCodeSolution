//
//  Solution+Hard.swift
//
//
//  Created by Wei on 2019/07/19.
//

//MARK:- Hard
public extension Solution{
    
    /**
     4. Median of Two Sorted Arrays
     */
    func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
        let m = nums1.count
        let n = nums2.count
        let medianIdx = (m + n) / 2
        
        var arr = Array(repeating: 0, count: m + n)
        var i = 0
        var j = 0
        for a in 0..<arr.count{
            if i == m{
                arr[a] = nums2[j]
                j += 1
            }
            else if j == n{
                arr[a] = nums1[i]
                i += 1
            }
            else if nums1[i] < nums2[j]{
                arr[a] = nums1[i]
                i += 1
            }else{
                arr[a] = nums2[j]
                j += 1
            }
        }
        
        if (m + n) % 2 == 0{
            return Double(arr[medianIdx] + arr[medianIdx-1]) / 2
        }else{
            return Double(arr[medianIdx])
        }
    }

    
    /**
     23. Merge k Sorted Lists
     */
    func Algorithm_mergeKLists(_ lists: [ListNode?]) -> ListNode? {
        
        func merge(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
            guard let l1 = l1 else { return l2 }
            guard let l2 = l2 else { return l1 }
            var head: ListNode
            if l1.val < l2.val {
                head = l1
                head.next = merge(head.next, l2)
            }
            else{
                head = l2
                head.next = merge(head.next, l1)
            }
            return head
        }
        
        
        if Bool.random(){
            /*
             //bad algorithm
             var head: ListNode?
             for i in lists.indices{
             head = merge(head, lists[i])
             }
             return head
             */
            //half faster algorithm
            var queue = lists.compactMap{ $0 }
            while queue.count > 1 {
                let q1 = queue.removeFirst()
                let q2 = queue.removeFirst()
                if let m  = merge(q1, q2){
                    queue.append(m)
                }
            }
            return queue.first
        }else{
            //mergeSort
            if lists.count <= 1{
                return lists.first ?? nil
            }
            let mid = lists.count / 2
            let left = Algorithm_mergeKLists(Array(lists[..<mid]))
            let right = Algorithm_mergeKLists(Array(lists[mid...]))
            return merge(left, right)
        }
    }

    /**
     97. Interleaving String
     */
    func isInterleave(_ s1: String, _ s2: String, _ s3: String) -> Bool {
        let s1 = Array(s1)
        let s2 = Array(s2)
        let s3 = Array(s3)
        guard s1.count + s2.count == s3.count else { return false }
        
        var dp = Array(repeating: Array(repeating: false, count: s2.count+1), count: s1.count+1)
        
        //compute
        for i in 1...s1.count {
            for j in 1...s2.count{
                if i == 0, j == 0{
                    dp[i][j] = true
                }
                else if i == 0 {
                    dp[i][j] = dp[i][j - 1] && s2[j - 1] == s3[i + j - 1]
                }
                else if j == 0 {
                    dp[i][j] = dp[i - 1][j] && s1[i - 1] == s3[j + i - 1]
                }
                else{
                    dp[i][j] = (dp[i - 1][j] && s1[i - 1] == s3[i + j - 1])
                        || (dp[i][j - 1] && s2[j - 1] == s3[i + j - 1])
                }
            }
        }
        return dp[s1.count][s2.count]
    }


    /**
     145. Binary Tree Postorder Traversal
     */
    func postorderTraversal(_ root: TreeNode?) -> [Int] {
        var stack = [root]
        var res: [Int] = []
        while !stack.isEmpty{
            guard let node = stack.removeLast() else { continue }
            res.append(node.val)
            
            if let l = node.left{
                stack.append(l)
            }
            if let r = node.right{
                stack.append(r)
            }
        }
        
        return res.reversed()
    }

    
    /**
     460. LFU Cache
     */
    class GenericLFUCache<Key: Hashable, Value: Any> {
        /*
         Author: Wei
         Note: I take time to figure out what LFUCache I want.
         
         Use Double List Node + HashMap
         set a property HashMap of [Key: DLListNode<Value>]
         set a property HashMap of [Count: DLList]
         
         Now we implement the following methods in DLListNode:
         * init(_ key: Key, _ value: Value)
         * remove()
         
         And then we implement the following methods:
         * init?(_ capacity: Int) {
         * get(Key) -> Value?
         * set(Key, Value) // put(Key, Value) //
         * sink(Node)
         * pop()
         * add(Node)
         */
        
        private typealias Node = DLListNode<Key, Value>
        
        private class List{
            var head: Node
            var tail: Node
            
            var isEmpty: Bool {
                return head.next === tail && tail.prev === head
            }
            
            init(_ node: Node) {
                self.head = Node()
                self.tail = Node()
                self.head.next = node
                node.prev = self.head
                node.next = self.tail
                self.tail.prev = node
            }
            
            func add(_ node: Node) {
                node.prev = head
                node.next = head.next
                head.next?.prev = node
                head.next = node
            }
            
            func remove(_ node: Node) {
                node.prev?.next = node.next
                node.next?.prev = node.prev
            }
            
            func removeLast() -> Node {
                let node = tail.prev!
                remove(node)
                return node
            }
        }
        
        private class DLListNode<Key: Hashable, Value: Any> {
            var prev: Node?
            var next: Node?
            
            var key: Key!
            var value: Value!
            var frequency: Int!
            
            init(){
                
            }
            init(_ key: Key, _ value: Value) {
                self.key = key
                self.value = value
                self.frequency = 1
            }
        }

        private let capacity: Int

        private var datum: [Key: Node] = [:]
        private var countMap: [Int: List] = [:]
        
        private var size: Int { return datum.count }
        private var min: Int = 0 // min reference count among all nodes in the cache


        init?(_ capacity: Int) {
            //If no capacity return nil
            guard capacity > 0 else { return nil }
            self.capacity = capacity
        }

        func get(_ key: Key) -> Value? {
            if let node = datum[key] {
                sink(node)
                return node.value
            } else {
                return nil
            }
        }

        func put(_ key: Key, _ value: Value) {
            set(key, value)
        }

        func set(_ key: Key, _ value: Value) {
            if let node = datum[key] {
                node.value = value
                sink(node)
            } else {
                if size == capacity {
                    pop()
                }
                let node = Node(key, value)
                add(node)
            }
        }

        private func sink(_ node: Node) {
            guard let list = countMap[node.frequency] else { return }
            list.remove(node)
            if list.isEmpty, min == node.frequency {
                min += 1
            }
            node.frequency += 1
            update(node)
        }

        private func pop() {
            let list = countMap[min]
            guard let node = list?.removeLast() else { return }
            datum[node.key] = nil
        }

        private func add(_ node: Node) {
            min = node.frequency
            datum[node.key] = node
            update(node)
        }

        private func update(_ node: Node){
            if let list = countMap[node.frequency] {
                list.add(node)
            }else {
                countMap[node.frequency] = List(node)
            }
        }
    }

    class LFUCache {
        var cache: GenericLFUCache<Int, Int>?
        init(_ capacity: Int) {
            cache = GenericLFUCache<Int, Int>(capacity)
        }
        
        func get(_ key: Int) -> Int {
            if let exist = cache?.get(key){
                return exist
            }else{
                return -1
            }
        }
        
        func put(_ key: Int, _ value: Int) {
            cache?.set(key, value)
        }
    }

    /**
     * Your LFUCache object will be instantiated and called as such:
     * let obj = LFUCache(capacity)
     * let ret_1: Int = obj.get(key)
     * obj.put(key, value)
     */

    /**
     765. Couples Holding Hands
     */
    func minSwapsCouples(_ row: [Int]) -> Int {
        var data = row
        var moved = 0
        
        repeat {
            guard let first = data.first, let i = data.firstIndex(where: { $0 == getCouple(first) }) else { break }
            if i > 1 {
                moved = moved + 1
                data[1] = data[1] ^ data[i]
                data[i] = data[1] ^ data[i]
                data[1] = data[1] ^ data[i]
            }
            data.removeFirst(2)
            
        } while data.count > 2
        
        return moved
    }
    
    func getCouple(_ a: Int) -> Int{
        return a % 2 == 0 ? a + 1 : a - 1
    }
    
    
    /**
     778. Swim in Rising Water
     */
    //Using Heap, refer to url: https://leetcode.com/submissions/detail/144698935/
    
    
    
    /**
     1147. Longest Chunked Palindrome Decomposition
     */
    func longestDecomposition_1(_ text: String) -> Int {
        let chars = Array(text)
        var l = chars.startIndex
        var r = chars.endIndex - 1
        let mid = (l + r + 1) / 2
        var ans = 0
        while l < r {
            let range = 0..<mid-l
            var next  = range.upperBound
            for i in range where chars[l...l+i] == chars[r-i...r] {
                next = i + 1
                ans += 2
                if l + i + 1 == r - i {
                    return ans
                }else{
                    break
                }
            }
            l += next
            r -= next
        }
        return ans + 1
    }
    func longestDecomposition(_ text: String) -> Int {
        let s = Array(text)
        var result = 0
        
        let end = s.endIndex - 1
        var i = s.startIndex
        var j = s.startIndex
        
        //ghiabcdefhelloadamhelloabcdefghi
        while i <= end{
            while j <= end {
                let range1 = i...(i+j)
                let range2 = (end-i-j)...(end-i)
                j += 1
                guard s[range1] == s[range2] else { continue }
                result += 1
                break
            }
            
            i += j
            j = 0
        }
        return result
    }

}

//
//  Solution+Hard.swift
//
//
//  Created by Wei on 2019/07/19.
//

//MARK:- Hard
public extension Solution{
    
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

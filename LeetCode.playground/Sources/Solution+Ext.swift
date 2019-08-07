import Foundation

extension Solution{
    class ListBuilder{
        static func get(_ nums: [Int]) -> ListNode? {
            let nodes = nums.map{ ListNode($0) }
            var head: ListNode?
            for i in nodes.indices{
                head?.next = nodes[i]
                head = nodes[i]
            }
            return nodes.first
        }
    }
    
    class TreeBuilder{
        static func get(_ nums: [Int?]) -> TreeNode? {
            var nums = nums
            var queue: [TreeNode] = []
            var root: TreeNode?
            
            if let rootVal = nums.removeFirst() {
                root = TreeNode(rootVal)
                queue.append(root!)
            }
            
            while !nums.isEmpty{
                for _ in queue.indices{
                    let q = queue.removeFirst()
                    
                    if nums.count > 0, let leftVal = nums.removeFirst() {
                        let left = TreeNode(leftVal)
                        q.left = left
                        queue.append(left)
                    }
                    
                    if nums.count > 0, let rightVal = nums.removeFirst() {
                        let right = TreeNode(rightVal)
                        q.right = right
                        queue.append(right)
                    }
                }
            }
            return root
        }
    }
    
}

import Foundation

typealias e844 = Solution
extension e844{
    /**
     844, Backspace String Compare
     
     Refer to [link](https://leetcode.com/problems/backspace-string-compare/)
     - Requires: Easy
     */
    public func backspaceCompare(_ S: String, _ T: String) -> Bool {
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
}




//
//  Solution.swift
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
     414, Third Maximum Number
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
     844, Backspace String Compare
     */
    func backspaceCompare(_ S: String, _ T: String) -> Bool {
        return removeBackspace(S) == removeBackspace(T)
    }
    private func removeBackspace(_ s: String) -> String{
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

let s = Solution()

print("Compiled")

//struct Iterator<Item>{
//    private var items: [Item] = []
//
//    private var index = -1
//
//    func hasNext(){
//        return index + 1 < Item.count
//    }
//
//    func next() -> Item{
//        index += 1
//        return items[index]
//    }
//
//    func append(_ val: Item){
//        self.items.append(val)
//    }
//}

//struct Convertor{
//    private var items: [Int] = []
//
//    var result: String
//
//    init(_ val: Int, type: Int){
//        var value = val
//        while value > 0{
//            items.append(value % type)
//            value = value / type
//        }
//        result = items.reversed().map{ String($0) }.joined()
//    }
//}
//
//let c = Convertor(50, type: 16)
//print(c.result)

let test = SortSolution.test()
print(test)

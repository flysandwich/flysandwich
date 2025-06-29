from ns import ns
ns.cppyy.cppdef("""

namespace ns3 {
// 定義一個函數：輸入兩個 Vector，輸出一個 Vector
Vector AddVectors(const Vector &a, const Vector &b) {
    Vector position = Vector(a.x + b.x, a.y + b.y, a.z + b.z);
    return position; 
}
}
""")

# 呼叫剛剛定義的 C++ 函數
v1 = ns.Vector(1.0, 2.0, 3.0)
v2 = ns.Vector(4.0, 5.0, 6.0)

# 使用剛剛定義的 C++ 函數：ns3::AddVectors
v3 = ns.AddVectors(v1, v2)
print("位置為", v3)
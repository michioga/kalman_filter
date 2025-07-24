## プログラムの概要

このRustプログラムは、非線形な状態を持つ移動体の位置を推定するためのシミュレーションです。具体的には、以下の4種類のカルマンフィルタを実装し、その性能を比較しています。

- 拡張カルマンフィルタ (EKF : Extended Kalman Filter)
- アンセンテッドカルマンフィルタ (UKF : Unscented Kalman Filter)
- キューバチャーカルマンフィルタ (CKF : Cubature Kalman Filter)
- ロバストキューバチャーカルマンフィルタ(SR-CKF/RCKF : Square-Root Cubature Kalman Filter)

シミュレーションでは、真の形状として円運動する物体を想定し、ノイズを含んだ観測地から各フィルタが物体の状態(位置、速度、進行方向)を、どの程度
正確に推定するかを検証します。

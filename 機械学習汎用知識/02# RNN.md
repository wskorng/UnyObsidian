## RNNの計算グラフ
RNN = Recurrent Neural Network
最も簡単な例のRNNの計算グラフを示します

```mermaid
flowchart TD

  %% 損失ノード
  subgraph .
    sum -->|L| loss
  end
  
  sigma0 -->|h0| mula1

  %% Layer1
  subgraph Layer1
    ParamW1["パラメータW"] -->|W| mula1((mul))
    mula1 -->|Wh0| adda1((add))
    X1["説明変数x1"] -->|x1| mulb1((mul))
    ParamU1["パラメータU"] -->|U| mulb1
    mulb1 -->|Ux1| adda1
    adda1 -->|Wh0+Ux1| sigma1((σ))
    ParamV1["パラメータV"] -->|V| mulc1
    sigma1 -->|h1| mulc1((mul))
    mulc1 -->|y1| sub1((sub))
    sub1 -->|y1-t1| sq1((square))
    sq1 -->|"(y1-t1)²"| scan1((scan))
    scan1 -->|L1| sum
  end

  sigma1 -->|h1| mula2

  %% Layer2
  subgraph Layer2
    ParamW2["パラメータW"] -->|W| mula2((mul))
    mula2 -->|Wh1| adda2((add))
    X2["説明変数x2"] -->|x2| mulb2((mul))
    ParamU2["パラメータU"] -->|U| mulb2
    mulb2 -->|Ux2| adda2
    adda2 -->|Wh1+Ux2| sigma2((σ))
    ParamV2["パラメータV"] -->|V| mulc2
    sigma2 -->|h2| mulc2((mul))
    mulc2 -->|y2| sub2((sub))
    sub2 -->|y2-t2| sq2((square))
    sq2 -->|"(y2-t2)²"| scan2((scan))
    scan2 -->|L2| sum
  end

  sigma2 -->|h2| mula3

  %% Layer3
  subgraph Layer3
    ParamW3["パラメータW"] -->|W| mula3((mul))
    mula3 -->|Wh2| adda3((add))
    X3["説明変数x3"] -->|x3| mulb3((mul))
    ParamU3["パラメータU"] -->|U| mulb3
    mulb3 -->|Ux3| adda3
    adda3 -->|Wh2+Ux3| sigma3((σ))
    ParamV3["パラメータV"] -->|V| mulc3
    sigma3 -->|h3| mulc3((mul))
    mulc3 -->|y3| sub3((sub))
    sub3 -->|y3-t3| sq3((square))
    sq3 -->|"(y3-t3)²"| scan3((scan))
    scan3 -->|L3| sum
  end

```


## RNNの誤差逆伝播のグラフ
```mermaid
flowchart TD

  %% 損失ノード
  subgraph .
    loss -->|1| sum
  end
  
  mula1 -->|@W| sigma0

  %% Layer1
  subgraph Layer1
    mula1((mul)) -->|⊗h0| ParamW1["パラメータW"]
    adda1((add)) -->|1| mula1
    mulb1((mul)) -->|@U| X1["説明変数x1"]
    mulb1 -->|⊗x1| ParamU1["パラメータU"]
    adda1 -->|1| mulb1
    sigma1((σ)) -->|σ'| adda1
    mulc1 -->|⊗h0| ParamV1["パラメータV"]
    mulc1((mul)) -->|@V| sigma1
    sub1((sub)) -->|1| mulc1
    sq1((square)) -->|"2(y1-t1)"| sub1
    scan1((scan)) -->|1| sq1
    sum -->|1| scan1
  end

  mula2 -->|@W| sigma1

  %% Layer2
  subgraph Layer2
    mula2((mul)) -->|⊗h0| ParamW2["パラメータW"]
    adda2((add)) -->|1| mula2
    mulb2((mul)) -->|@U| X2["説明変数x2"]
    mulb2 -->|⊗x2| ParamU2["パラメータU"]
    adda2 -->|1| mulb2
    sigma2((σ)) -->|σ'| adda2
    mulc2 -->|⊗h0| ParamV2["パラメータV"]
    mulc2((mul)) -->|@V| sigma2
    sub2((sub)) -->|1| mulc2
    sq2((square)) -->|"2(y2-t2)"| sub2
    scan2((scan)) -->|1| sq2
    sum -->|1| scan2
  end

  mula3 -->|@W| sigma2

  %% Layer3
  subgraph Layer3
    mula3((mul)) -->|⊗h0| ParamW3["パラメータW"]
    adda3((add)) -->|1| mula3
    mulb3((mul)) -->|@U| X3["説明変数x3"]
    mulb3 -->|⊗x3| ParamU3["パラメータU"]
    adda3 -->|1| mulb3
    sigma3((σ)) -->|σ'| adda3
    mulc3 -->|⊗h0| ParamV3["パラメータV"]
    mulc3((mul)) -->|@V| sigma3
    sub3((sub)) -->|1| mulc3
    sq3((square)) -->|"2(y3-t3)"| sub3
    scan3((scan)) -->|1| sq3
    sum -->|1| scan3
  end

```

## RNNの各パラメータの誤差逆伝播の計算式
したがって、以下の勾配が得られます：

$$
\begin{align*}
\frac{\partial L}{\partial V} &= 2(y_1-t_1)\otimes h_1 + 2(y_2-t_2)\otimes h_2 + 2(y_3-t_3)\otimes h_3 + \cdots \\
\frac{\partial L}{\partial (Wh_0+Ux_1)} &= 2(y_1-t_1)\circledast V*\sigma'_1 + 2(y_2-t_2)\circledast V*\sigma'_2\circledast W*\sigma'_1 + 2(y_3-t_3)\circledast V*\sigma'_3\circledast W*\sigma'_2\circledast W*\sigma'_1 + \cdots \\
\frac{\partial L}{\partial W} &= \left(\frac{\partial L}{\partial (Wh_0+Ux_1)}\right)\otimes h_0 + \left(\frac{\partial L}{\partial (Wh_1+Ux_2)}\right)\otimes h_1 + \left(\frac{\partial L}{\partial (Wh_2+Ux_3)}\right)\otimes h_2 + \cdots \\
\frac{\partial L}{\partial U} &= \left(\frac{\partial L}{\partial (Wh_0+Ux_1)}\right)\otimes x_1 + \left(\frac{\partial L}{\partial (Wh_1+Ux_2)}\right)\otimes x_2 + \left(\frac{\partial L}{\partial (Wh_2+Ux_3)}\right)\otimes x_3 + \cdots
\end{align*}
$$

ここで、$\otimes$ は直積を、$*$ はアダマール積を、$\circledast$ は行列積を表します。

大まかに$\sigma'$を$1$とみなして考えると、これがやばいことがわかる。
というのも$\frac{\partial L}{\partial (Wh_0+Ux_1)}$の第$t$項は$W^t$かかってる。もし$||W||>1$とすると古い項ほど影響が大きくなるので、そんなわけなくて$||W||<1$なんだけど、それはそれで古い項の影響がべき乗で小さくなりすぎる、例えば$||W||\sim0.8$くらいでも$t\sim20$くらいで$||W^t||\sim0.01$となりほぼ覚えてないことになる。(これを別の切り口で言うと勾配消失問題にあたるのかな？)
これが理由でシンプルなRNNでは10～20項くらいで記憶力の限界になる。
この問題の原因について、人間の記憶力って短期記憶(脳科学的な短期記憶に等しいとは限らない。言語モデルでいえばさっき書いた主語が男性名詞だから述語の性も男性にしなきゃレベルの短期)はべき乗で減衰していいけど長期記憶はそうじゃないよねってことで開発されたのがLSTM。


## RNNセル
```mermaid
flowchart TD
  sigma0(sigma0) -->|h0| mula1

  %% Layer1
  subgraph Layer1
    ParamW1["パラメータW"] -->|W| mula1((mul))
    mula1 -->|Wh0| adda1((add))
    X1["説明変数x1"] -->|x1| mulb1((mul))
    ParamU1["パラメータU"] -->|U| mulb1
    mulb1 -->|Ux1| adda1
    adda1 -->|Wh0+Ux1| sigma1((σ))
    sigma1 -->|h1| out1
  end

  sigma1 -->|h1| mula2((mul))
```
の部分をRNNセルといい、
```mermaid
flowchart TD
  gate0((gate σ W U)) -->|h0| gate1

  %% Layer1
  subgraph Layer1
    X1["説明変数x1"] -->|x1| gate1((gate σ W U))
    gate1 -->|h1| out1
  end

  gate1 -->|h1| gate2((gate σ W U))
```
と略します。これの逆伝播グラフは以下のようになります。
```mermaid
flowchart TD
  gate2((gate σ W U)) -->|σ'@W| gate1

  %% Layer1
  subgraph Layer1
    gate1((gate σ W U)) -->|σ'@U| X1["説明変数x1"]
    out1 -->|"...@V"| gate1
  end

  gate1 -->|σ'@W| gate0((gate σ W U))

```

## LSTMセル
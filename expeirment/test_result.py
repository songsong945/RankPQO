import re

data = """
Test results for 30c
train loss: 0.3740527600242446
train accuracy: 0.6195833333333334
validation loss: 0.3728869123359521
validation accuracy: 0.63
test loss: 0.3702379853328069
test accuracy: 0.6366666666666667
Test results for 16c
train loss: 1.5655350194871425
train accuracy: 0.5704166666666667
validation loss: 1.5389358305931091
validation accuracy: 0.5816666666666667
test loss: 1.4868637832005818
test accuracy: 0.6066666666666667
Test results for 21c
train loss: 0.6944308016449213
train accuracy: 0.7033333333333334
validation loss: 0.637489579518636
validation accuracy: 0.6933333333333334
test loss: 0.5427869572242101
test accuracy: 0.7433333333333333
Test results for 9a
train loss: 0.8278030919581651
train accuracy: 0.514
validation loss: 0.8309830437898635
validation accuracy: 0.508
test loss: 0.8727575154304504
test accuracy: 0.488
Test results for 13b
train loss: 1.156519239001597
train accuracy: 0.666875
validation loss: 1.3583111143112183
validation accuracy: 0.6333333333333333
test loss: 1.196978627840678
test accuracy: 0.655
Test results for 13c
train loss: 0.24804911735157173
train accuracy: 0.49875
validation loss: 0.21010348598162334
validation accuracy: 0.52
test loss: 0.248022202650706
test accuracy: 0.5233333333333333
Test results for 15a
train loss: 0.3630565776924292
train accuracy: 0.4945833333333333
validation loss: 0.48457093318303424
validation accuracy: 0.49666666666666665
test loss: 0.3512260267138481
test accuracy: 0.5066666666666667
Test results for 4a
train loss: 0.4255439733245887
train accuracy: 0.7698125
validation loss: 0.42257427395155084
validation accuracy: 0.767
test loss: 0.4489766331605392
test accuracy: 0.761
Test results for 23b
train loss: 0.6497988847121596
train accuracy: 0.5039375
validation loss: 0.6737879102230072
validation accuracy: 0.494
test loss: 0.6588581656217575
test accuracy: 0.5165
Test results for 33a
train loss: 0.1575920034367591
train accuracy: 0.6133125
validation loss: 0.17321995317935943
validation accuracy: 0.59
test loss: 0.14515954300016165
test accuracy: 0.623
Test results for 27b
train loss: 0.19926988999545575
train accuracy: 0.5366875
validation loss: 0.211535272359848
validation accuracy: 0.532
test loss: 0.20271531036496163
test accuracy: 0.5405
Test results for 2b
train loss: 0.17302539832890035
train accuracy: 0.50875
validation loss: 0.16219603419303893
validation accuracy: 0.59
test loss: 0.20461201548576355
test accuracy: 0.51
Test results for 31a
train loss: 0.18411487221717834
train accuracy: 0.49875
validation loss: 0.2224808144569397
validation accuracy: 0.52
test loss: 0.19481462121009827
test accuracy: 0.56
"""

# 使用正则表达式提取所有需要的数据
matches = re.findall(
    r"Test results for (\w+)\ntrain loss: [\d.]+\ntrain accuracy: ([\d.]+)\nvalidation loss: [\d.]+\nvalidation accuracy: ([\d.]+)\ntest loss: [\d.]+\ntest accuracy: ([\d.]+)",
    data)

print(",".join(match[0] for match in matches))
print(",".join(match[1] for match in matches))
print(",".join(match[2] for match in matches))
print(",".join(match[3] for match in matches))

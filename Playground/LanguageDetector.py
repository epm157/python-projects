from langdetect import detect
from langdetect import detect_langs

from langdetect import DetectorFactory
DetectorFactory.seed = 0

sent1 = "War doesn't show who's right, just who's left."
sent2 = "وَلاَ تَكُونَنَّ مِنَ الْمُشْرِكِينَ"

print(detect_langs(sent2))
print(detect(sent2))


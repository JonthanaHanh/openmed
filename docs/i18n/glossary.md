# Hindi README glossary

`README.hi.md` को अपडेट करते समय इन canonical renderings का उपयोग करें। Product,
library, framework, model और protocol के आधिकारिक नाम उसी रूप में रखें जब अनुवाद
से उनकी पहचान कठिन हो सकती है।

| English term | Canonical Hindi | Usage note |
|---|---|---|
| de-identification | डी-आइडेंटिफिकेशन | पहचान करने वाली जानकारी हटाने या बदलने की प्रक्रिया के लिए। |
| de-identified | डी-आइडेंटिफाइड | डी-आइडेंटिफिकेशन के बाद के output के लिए। |
| on-device | डिवाइस पर | जब computation उपयोगकर्ता के device पर होता है। |
| entity extraction | एंटिटी निष्कर्षण | Biomedical, clinical या PII entities निकालने के लिए। |
| local-first | लोकल-फर्स्ट | Architecture और product principle के लिए। |
| Privacy Filter | Privacy Filter | Official model-family name में English रूप बनाए रखें। |
| PII detection | PII पहचान | नए पाठक के लिए आवश्यक होने पर पहली बार PII का विस्तार करें। |
| redaction | PII छिपाना | Sensitive spans को mask करने या हटाने के लिए। |
| model-backed PII language | मॉडल-समर्थित PII भाषा | इसे validator-only national-ID coverage से अलग रखें। |
| smart entity merging | स्मार्ट एंटिटी मर्जिंग | Fragmented token spans को फिर से जोड़ने के लिए। |
| air-gapped | एयर-गैप्ड | पहली बार उपयोग पर English term कोष्ठक में दिया जा सकता है। |
| clinical NER | क्लिनिकल NER | NER को code और model identifiers में English में रखें। |
| vendor lock-in | वेंडर लॉक-इन | OpenMed के आश्वासन के लिए `कोई वेंडर लॉक-इन नहीं` लिखें। |
| checkpoint | चेकपॉइंट | Published model checkpoint के लिए। |

Translation update की समीक्षा के बाद section manifest को refresh और verify करें:

```bash
python scripts/i18n/check_readme_drift.py --update
python scripts/i18n/check_readme_drift.py
```

## Naming Rule of detection result

```$(STATUS)_$(LENGTH)m_$(DETECTION)_con$(CONSECTIVE)_w$(WINDOW)$(INFO).txt```

- STATUS: status of the user, *concentrate* or *distract*
- LENGTH: minute of the video, an integer
- DETECTION: *normal* or *face*
- CONSECTIVE: the sufficient number of consecutive frames, usually *2* or *3*
- WINDOW: the size of dynamic window
- INFO: extra information to describe the dynamic threshold

e.g., concentrate_3m_normal_con3_w1500.txt

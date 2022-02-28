## Naming Rule of detection result

```$(STATUS)_$(LENGTH)m_$(DETECTION)_con$(CONSECTIVE)_w$(WINDOW).txt```

- STATUS: status of the user, *concentrate* or *disconcentrate*
- LENGTH: minute of the video, an integer
- DETECTION: *normal* or *face*
- CONSECTIVE: the sufficient number of consecutive frames, usually *2* or *3*
- WINDOW: the size of dynamic window

e.g., concentrate_3m_normal_con3_w1500.txt

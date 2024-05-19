# Master’s thesis - Object detection and segmentation in historical encrypted manuscripts

Slovak University of Technology in Bratislava - Faculty of Electrical Engineering and Information Technology (FEI STU)

## Abstract
**Study Programme:** Applied Informatics

**Author:** Bc. Filip Mikuš

**Master’s thesis:** Object detection and segmentation in historical encrypted manuscripts

**Supervisor:** Ing. Pavol Marák, PhD.

**Place and year of submission:** Bratislava 2024

With the increasing number of digitized historical encrypted documents available in archives, the development of computing hardware and machine learning (ML), efforts to automate the processing of historical encrypted documents are intensifying. This process speeds up the work with historical sources by applying automated transcription using detection and segmentation of symbols by ML models. 

The aim of our work was to study the form of historical encrypted manuscripts, to document, train and test state-of-the-art detection/segmentation models or techniques, and integrate them into a web application programming interface (API). 

Based on the collected theoretical knowledge, we trained and tested the detection/segmentation models YOLOv8, YOLOv9, YOLO-NAS, YOLO-Worldv2, RT-DETR and FastSAM on the provided datasets of digits and symbols (glyphs). We have used the SAHI technique (Slicing Aided Hyper Inference) to improve the detection of small objects. The trained models were exported to the universal ONNX format, whose inference times were compared with the native PyTorch export format. For automated transcription of the detected symbols, we designed and implemented a custom transcription mechanism using unsupervised learning (UL). We integrated the trained exported models and the mechanism for automated transcription using UL, along with tools for automated annotation generation using ML and for advanced dataset browsing using ML, into a REST web API that we designed and implemented. We tested the API using HTTP tests and performance tests.

**Keywords:** historical encrypted manuscripts, nomenclators, detection, transcription, artificial intelligence, YOLO, DETR, SAM, SAHI, ONNX, web REST API, segmentation, digits, glyphs, machine learning

---

# Diplomová práca - Detekcia a segmentácia objektov v historických šifrovaných rukopisoch

Slovenská technická univerzita v Bratislave - Fakulta elektrotechniky a informatiky (FEI STU)

## Súhrn
**Študíjny program:** Aplikovaná informatika

**Autor:** Bc. Filip Mikuš

**Diplomová práca:** Detekcia a segmentácia objektov v historických šifrovaných rukopisoch

**Vedúci záverečnej práce:** Ing. Pavol Marák, PhD.

**Miesto a rok predloženia práce:** Bratislava 2024

S narastajúcim počtom dostupných digitalizovaných historických dokumentov v archívoch, vývojom výpočtového hardvéru a modelov strojového učenia (ML) sa zintenzívňuje snaha o automatizované spracovanie historických šifrovaných dokumentov. Tento proces prácu s historickými prameňmi urýchľuje aplikovaním automatizovanej transkripcie a detekcie/segmentácie symbolov ML modelmi. 

Cieľom tejto práce bolo oboznámiť sa s podobou historických šifrovaných rukopisov, zdokumentovať, dotrénovať a otestovať najmodernejšie detekčné/segmentačné modely, respektíve techniky a zahrnúť ich do webového aplikačného rozhrania (API). 

Na základe získaných poznatkov sme na poskytnutých datasetoch číslic a symbolov (glyfov) natrénovali a otestovali modely YOLOv8, YOLOv9, YOLO-NAS, YOLO-Worldv2, RT-DETR a FastSAM. Na zlepšenie detekcie malých objektov sme využili techniku SAHI (Slicing Aided Hyper Inference). Natrénované modely boli exportované do univerzálneho ONNX formátu, ktorého inferenčné časy boli porovnané s natívnym exportným formátom PyTorch. Na automatizovanú transkripciu detegovaných symbolov sme navrhli a implementovali vlastný transkripčný mechanizmus s využitím strojového učenia bez učiteľa (UL). Natrénované vyexportované modely a mechanizmus na automatizovanú transkripciu s využitím UL sme spolu s nástrojmi na automatizované generovanie anotácií s využitím ML a na pokročilé prehľadávanie datasetov s využitím ML zahrnuli do nami navrhnutého a implementovaného webového API typu REST. Výsledné API sme otestovali pomocou HTTP testov a záťažových (performance) testov.

**Kľúčové slová:** historické šifrované rukopisy, nomenklátory, detekcia, segmentácia, transkripcia, čislice, glyfy, umelá inteligencia, strojové učenie, YOLO, DETR, SAM, SAHI, ONNX, webové REST API

---


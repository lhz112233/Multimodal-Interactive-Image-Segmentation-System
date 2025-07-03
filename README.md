# Multimodal-Interactive-Image-Segmentation-System
This is my first project on GitHub. It is an outstanding undergraduate thesis.
### 1 System Functional Requirements

This system is designed to provide users with an efficient and convenient solution for image segmentation and analysis.It implements a multimodal interactive image segmentation system based on SAM(Segment Anything Model)for image segmentation and CLIP for cross-modal understanding.To ensure the completeness and ease of operation of the system,the following functions have been designed:


• Basic Interaction Functions:Support for batch processing of multiple locally uploaded images and flexible switching to real-time camera preview,with dynamic thumbnail previews and instant feedback on image information.


• Intelligent Processing Core:Implementation of pixel-level automatic mask generation based on SAM,optimization of region matching or merging through a dual-dimensional control panel,and support for adjustable transparency of the segmentation overlay layer.


• Cross-Modal Retrieval System:Utilization of CLIP for text-image feature matching,support for multi-keyword joint retrieval and adjustable similarity threshold,with automatic highlighting of matched regions.


• Interactive Parameter Tuning:Provision of a dual-dimensional control panel(matching threshold/region merging threshold)to optimize segmentation results through parameter adjustment.


• Full-Process Monitoring System:Construction of a three-level log tracking system(single-image details/batch statistics/history records)and generation of visualized matching details and timing statistics tables.


• Artificial Intelligence Assistance:Integration of the DeepSeek large model to provide parameter tuning suggestions and semantic interpretation of results,with context-aware conversational operation guidance.


• Asynchronous Processing Architecture:Implementation of safe process termination and resumable interrupted processing,optimization of asynchronous batch processing queues through dynamic resource scheduling to improve multi-task concurrency efficiency.

### 2 System Overall Architecture

The overall framework of the system is shown in Figure 4.1.Implemented in Python,the multimodal interactive image segmentation system is divided into front-end and back-end components:Gradio and FastAPI.The back-end uses the FastAPI framework to integrate CLIP and SAM models,implementing functions such as text-image retrieval,image segmentation,and AI assistance.The front-end employs a lightweight Gradio framework to build an intuitive upload interface.Inheriting the multimodal(image,text)concept from the CLIP model,a dual-modal user interface is designed.The"Image Segmentation"module supports multiple image upload methods,text description input,parameter settings,and result display.The"AI Assistant"module,with the help of the DeepSeek large model,supports intelligent Q&A(such as translation and popular science)and operation guidance.Meanwhile,the front-end communicates with the back-end API through axios,facilitating interface calls in the development environment.

【photo1】

### 3 Interaction Interface Design

Based on the functions and modules of the system design,the final interactive Web UI design is shown in the figure.

【photo2】

【photo3】

The main interface implements components such as the navigation bar for switching module interfaces,image upload,text description input,image preview,output content display area,and multi-button operating system area,as shown in the table.

【table1】

import cv2 as cv


def pause():
    cap.release()
    print("teste")
# # ------------------MANIPULANDO VIDEO/FRAMES-----------------q
# if key == ord('j'):
#     # cv.destroyWindow('tela')
#     # cv.destroyWindow('tela2')
#     # cv.destroyWindow('tela3')
#     cv.destroyWindow('hori')
#     frame3 = np.zeros((255, 255, 3), np.uint8)
#     cv.imshow('Frame3', frameArray[0])
#     counter = 0
#     i = 0
#
#     while key != ord('q'):
#         # -------------START/PAUSE------------------
#         if key == ord('p'):
#             while True:
#                 if i > 0:
#                     for i in np.arange(i, len(frameArray)):
#                         cv.imshow('Frame3', frameArray[i])
#                         key = cv.waitKey(50)
#                         if key == ord('r'):
#                             i = 0
#                             break
#                         if key == ord('p'):
#                             break
#                 else:
#                     for i in np.arange(1, len(frameArray)):
#                         cv.imshow('Frame3', frameArray[i])
#                         key = cv.waitKey(50)
#                         if key == ord('r'):
#                             i = 0
#                             break
#                         if key == ord('p'):
#                             break
#                 break
#
#         # -----------AVANÇAR FRAME-------------
#         if key == ord('l'):
#             if i < len(frameArray) - 1:
#                 i = 1 + i
#                 cv.imshow('Frame3', frameArray[i])
#
#         # ----------VOLTAR FRAME--------------
#         if key == ord('k'):
#             if i > 0:
#                 i = i - 1
#                 cv.imshow('Frame3', frameArray[i])
#
#         # ----------RESTART------------------
#         if key == ord('r'):
#             i = 0
#             cv.imshow('Frame3', frameArray[i])
#
#         key = cv.waitKey(1)
#         if key == ord('q'):
#             break
#     cv.destroyAllWindows()

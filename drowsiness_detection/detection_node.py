import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import time
import os

class DrowsinessNode(Node):
    def __init__(self):
        super().__init__('drowsiness_node')
        # Substitua '/image_raw' pelo tópico correto da sua câmera se necessário
        self.subscription = self.create_subscription(
            Image,
            '/image_raw', 
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        
        # Carrega o modelo (garanta que o best.pt está na mesma pasta onde você roda o nó)
        self.model = YOLO('drowse_model.pt') 
        
        # Variáveis do Cronômetro
        self.inicio_fechado = None
        self.alarme_tocando = False

    def listener_callback(self, msg):
        # 1. Converte a imagem do ROS para OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. Roda o YOLO na imagem
        results = self.model(frame, conf=0.5, verbose=False)
        
        olho_fechado = False
        olho_aberto = False
        
        # 3. Analisa o que o YOLO encontrou neste frame
        for box in results[0].boxes:
            nome_classe = self.model.names[int(box.cls)]
            if nome_classe == 'closed':
                olho_fechado = True
            elif nome_classe == 'open':
                olho_aberto = True
        
        # 4. Lógica de Tempo e Alarme
        if olho_fechado and not olho_aberto:
            # Se acabou de fechar, marca a hora no relógio
            if self.inicio_fechado is None:
                self.inicio_fechado = time.time()
            else:
                # Calcula quanto tempo passou
                tempo_decorrido = time.time() - self.inicio_fechado
                
                # Passou de 5 segundos e o alarme ainda não tocou?
                if tempo_decorrido >= 2.0 and not self.alarme_tocando:
                    self.get_logger().warn("🚨 ALARME: MOTORISTA DORMINDO! 🚨")
                    
                    # O '&' no final faz o som rodar em segundo plano sem travar o vídeo
                    os.system("play -nq -t alsa synth 1.5 sine 880 &")
                    
                    self.alarme_tocando = True
                    
        elif olho_aberto:
            # Se o olho abriu, reseta o cronômetro e desativa o alarme
            self.inicio_fechado = None
            self.alarme_tocando = False
            
        # 5. Mostra a imagem com as caixas na tela
        annotated_frame = results[0].plot()
        cv2.imshow("Sistema Anti-Sono", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DrowsinessNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

package ufpr.gestaodainformacao.reconhecimento;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

public class Deteccao {

	private String caminhoDetector = "Recursos//haarcascade-frontalface-alt.xml";
	private CascadeClassifier detector;

	public Mat tratamentoImagem(Mat foto) {
		
		this.detector = new CascadeClassifier(this.caminhoDetector);
		RectVector vetor = new RectVector();
		this.detector.detectMultiScale(foto, vetor, 1.1, 1, 0, new Size(0,0), new Size(3000,3000));
		
		Rect retangulo = vetor.get(0);
		Mat retanguloFace = new Mat(foto, retangulo);
		resize(retanguloFace, retanguloFace, new Size(160,160));
		cvtColor(retanguloFace, retanguloFace, COLOR_BGRA2GRAY);
		
		return retanguloFace;
		
	}

}

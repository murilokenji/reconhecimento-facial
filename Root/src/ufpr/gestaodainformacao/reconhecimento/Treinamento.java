package ufpr.gestaodainformacao.reconhecimento;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import java.io.File;
import java.nio.IntBuffer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;

public class Treinamento {
	
	private File diretorioFotos;
	private String caminhoFotos = "Fotos//Treinamento";
	private Deteccao detector;
	private Mat rotulos;
	private MatVector vetorFotos;
	
	public Treinamento() {
		this.detector = new Deteccao();
	}
	
	public void treinamentoEigenfaces() {
		this.processamentoFotos();
		FaceRecognizer eigenfaces = createEigenFaceRecognizer(30, 0);
		eigenfaces.train(this.vetorFotos, this.rotulos);
		eigenfaces.save("Recursos//classificadorEigenFaces.yml");
	}
	
	public void treinamentoFisherfaces() {
		this.processamentoFotos();
		FaceRecognizer fisherfaces = createFisherFaceRecognizer(30, 0);
		fisherfaces.train(this.vetorFotos, this.rotulos);
		fisherfaces.save("Recursos//classificadorFisherFaces.yml");
	}
	
	public void treinamentoLBPH() {
		this.processamentoFotos();
		FaceRecognizer lbph = createLBPHFaceRecognizer(10, 10, 15, 15, 0);
		lbph.train(this.vetorFotos, this.rotulos);
		lbph.save("Recursos//classificadorLBPH.yml");
	}
	
	private void processamentoFotos(){
		this.diretorioFotos = new File(this.caminhoFotos);
		File[] arquivos = this.diretorioFotos.listFiles();
		
		this.vetorFotos = new MatVector(arquivos.length);
		this.rotulos = new Mat(arquivos.length, 1, CV_32SC1); 
		IntBuffer rotulosBuffer = rotulos.createBuffer();
		int contador = 0;
				
		for(File arquivo: arquivos) {
			
			Mat foto = imread(arquivo.getPath());
			resize(foto, foto, new Size(160,160));
			Mat fotoTratada = this.detector.tratamentoImagem(foto);
			
			int classe = Integer.parseInt(arquivo.getName().split("\\.")[0]);
			
			this.vetorFotos.put(contador,fotoTratada);
			rotulosBuffer.put(contador, classe);
			contador++;
			
		}		
	}
	
	public static void main(String[] args) {
		Treinamento t = new Treinamento();
		t.treinamentoEigenfaces();
		t.treinamentoFisherfaces();
		t.treinamentoLBPH();
	}
}

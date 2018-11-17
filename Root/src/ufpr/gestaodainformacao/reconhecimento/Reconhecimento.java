package ufpr.gestaodainformacao.reconhecimento;

import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;

public class Reconhecimento {

	private String diretorioFoto = "Fotos//Reconhecimento";
	private String caminhoReconhecedor = "Recursos//";
	private Deteccao detector;

	public Reconhecimento() {
		this.detector = new Deteccao();
	}

	public void reconhecerEigenFaces() throws IOException {

		File dir = new File(this.diretorioFoto);
		File[] arquivos = dir.listFiles();

		FaceRecognizer reconhecedor = createEigenFaceRecognizer();
		reconhecedor.load(this.caminhoReconhecedor + "classificadorEigenFaces.yml");

		BufferedWriter buffWrite = new BufferedWriter(new FileWriter("resultadoEigenfaces.txt"));
		String titulo = "--RESULTADO--";
		buffWrite.append(titulo);

		for (File arquivo : arquivos) {

			int classe = Integer.parseInt(arquivo.getName().split("\\.")[0]);
			Mat foto = imread(arquivo.getAbsolutePath());
			Mat fotoTratada = this.detector.tratamentoImagem(foto);

			IntPointer rotulo = new IntPointer(1);
			DoublePointer confianca = new DoublePointer(1);

			reconhecedor.predict(fotoTratada, rotulo, confianca);
			int predicao = rotulo.get(0);

			if (predicao == classe) {
				buffWrite.newLine();
				buffWrite.append("Eigenfaces => Foto: " + arquivo.getName() + " => Reconhecido => Confiança : "
						+ confianca.get(0));
			} else {
				buffWrite.newLine();
				buffWrite.append("Eigenfaces => Foto: " + arquivo.getName() + " => Não Reconhecido");
			}

		}
		buffWrite.close();
	}

	public void reconhecerFisherFaces() throws IOException {

		File dir = new File(this.diretorioFoto);
		File[] arquivos = dir.listFiles();

		FaceRecognizer reconhecedor = createFisherFaceRecognizer();
		reconhecedor.load(this.caminhoReconhecedor + "classificadorFisherFaces.yml");

		BufferedWriter buffWrite = new BufferedWriter(new FileWriter("resultadoFisherfaces.txt"));
		String titulo = "--RESULTADO--";
		buffWrite.append(titulo);

		for (File arquivo : arquivos) {

			int classe = Integer.parseInt(arquivo.getName().split("\\.")[0]);
			Mat foto = imread(arquivo.getAbsolutePath());
			Mat fotoTratada = this.detector.tratamentoImagem(foto);

			IntPointer rotulo = new IntPointer(1);
			DoublePointer confianca = new DoublePointer(1);

			reconhecedor.predict(fotoTratada, rotulo, confianca);
			int predicao = rotulo.get(0);

			if (predicao == classe) {
				buffWrite.newLine();
				buffWrite.append("Fisherfaces => Foto: " + arquivo.getName() + " => Reconhecido => Confiança : "
						+ confianca.get(0));
			} else {
				buffWrite.newLine();
				buffWrite.append("Fisherfaces => Foto: " + arquivo.getName() + " => Não Reconhecido");
			}

		}
		buffWrite.close();
	}
	
	public void reconhecerLBPH() throws IOException {

		File dir = new File(this.diretorioFoto);
		File[] arquivos = dir.listFiles();

		FaceRecognizer reconhecedor = createLBPHFaceRecognizer();
		reconhecedor.load(this.caminhoReconhecedor + "classificadorLBPH.yml");

		BufferedWriter buffWrite = new BufferedWriter(new FileWriter("resultadoLBPH.txt"));
		String titulo = "--RESULTADO--";
		buffWrite.append(titulo);

		for (File arquivo : arquivos) {

			int classe = Integer.parseInt(arquivo.getName().split("\\.")[0]);
			Mat foto = imread(arquivo.getAbsolutePath());
			Mat fotoTratada = this.detector.tratamentoImagem(foto);

			IntPointer rotulo = new IntPointer(1);
			DoublePointer confianca = new DoublePointer(1);

			reconhecedor.predict(fotoTratada, rotulo, confianca);
			int predicao = rotulo.get(0);

			if (predicao == classe) {
				buffWrite.newLine();
				buffWrite.append("LBPH => Foto: " + arquivo.getName() + " => Reconhecido => Confiança : " + confianca.get(0));
			} else {
				buffWrite.newLine();
				buffWrite.append("LBPH => Foto: " + arquivo.getName() + " => Não Reconhecido");
			}

		}
		buffWrite.close();
	}
	
	public static void main(String[] args) throws IOException {
		Reconhecimento r = new Reconhecimento();
		r.reconhecerEigenFaces();
		r.reconhecerFisherFaces();
		r.reconhecerLBPH();
	}

}

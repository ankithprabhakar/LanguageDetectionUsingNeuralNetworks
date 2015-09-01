import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

public class WikiLangDetection {
	private static final int INPUT = 3;
	private static final int OUTPUT = 3;
	private static final int HIDDEN = 5;
	private static final int SAMPLES = 336;
	private static double[][] weightsHiddenTest = {{4.4865873319047065, -1.2880429753586928, -0.8507089411698147, 4.515693660720222, -0.016998167824425613}, {-5.140080831982334, 2.8924287972147313, 2.6373897101831743, -6.115183436696522, -2.316972948073841}, {-2.3821562396561857, -13.663416835724869, 4.3383835141270275, -0.8558126096244969, 12.107637655466503}, {0.03790661069388479, 0.836695370720703, 0.183367749742416, 0.494774138008937, 0.09415283397195606}};
	private static double[][] weightOutputTest =  {{8.722764716148697, -4.509751217000615, -5.934795920148496}, {-2.899196484733009, 11.125596402608291, -11.593898587893376}, {-5.625301037975154, -1.0143523402255512, 2.4367085202684753}, {9.91248113866473, -6.257610369591651, -5.156142766072712}, {-1.698492435256581, -9.70241556083725, 8.399818680230496}, {0.9435425016116548, 0.5570127131966174, 0.47504968116250645}};
	private static double[][] weightsHidden = new double[INPUT+1][HIDDEN];
	private static double[][] weightOutput = new double[HIDDEN+1][OUTPUT];

	private static double inputNeurons[] = new double[INPUT];
	private static double hiddenNeurons[] = new double[HIDDEN];
	private static double outputNeurons[] = new double[OUTPUT];
	private static double expectedNeurons[] = new double[OUTPUT];

	private static double inputNeuronsTest[] = new double[INPUT];
	private static double hiddenNeuronsTest[] = new double[HIDDEN];
	private static double outputNeuronsTest[] = new double[OUTPUT];

	private static int number =0;

	private static double deltaJ[] = new double[OUTPUT];
	private static double deltaI[] = new double[HIDDEN];
	private static double hiddenIn = 0.0;
	private static double outputIn = 0.0;
	private static double hiddenInTest = 0.0;
	private static double outputInTest = 0.0;

	private static double englishTrain[] = new double[200];
	private static double italianTrain[] = new double[200];
	private static double dutchTrain[] = new double[200];
	private static double englishTest = 0.0;
	private static double italianTest = 0.0;
	private static double dutchTest = 0.0;

	private static int localEnglishCount = 0;
	private static int localItalianCount = 0;
	private static int localDutchCount = 0;


	private static void backPropLearning(){
		assignRandomWeights();
		for(int xCount =0;xCount<2500;xCount++){
			localDutchCount = localEnglishCount = localItalianCount =0;
			for(int count =0;count<SAMPLES-2;count++){
				setInputAndOutputNeurons();
				number++;
				propagateInputsForward();
				propagateDeltasBackward();
			}
		}
	}

	private static void propagateDeltasBackward() {
		for(int i=0;i<OUTPUT;i++){
			deltaJ[i] = (outputNeurons[i]*(1.0-outputNeurons[i]))*(expectedNeurons[i]-outputNeurons[i]);
		}
		for(int i=0;i<OUTPUT;i++){
			for(int j=0;j<HIDDEN;j++){
				weightOutput[j][i]+=deltaJ[i]*hiddenNeurons[j]*1.5;
			}
		}
		for(int i=0;i<HIDDEN;i++){
			deltaI[i] = 0.0;
			for(int j=0;j<OUTPUT;j++){
				deltaI[i]+=deltaJ[j]*weightOutput[i][j];
			}
			deltaI[i] *= hiddenNeurons[i]*(1.0-hiddenNeurons[i]);
		}
		for(int i=0;i<HIDDEN;i++){
			for(int j=0;j<INPUT;j++){
				weightsHidden[j][i]+=deltaI[i]*inputNeurons[j]*1.5;
			}
		}
	}

	private static void propagateInputsForward() {
		for(int i=0;i<HIDDEN;i++){
			hiddenIn = 0.0;
			for(int j=0;j<INPUT;j++){
				hiddenIn+=weightsHidden[j][i]*inputNeurons[j];
			}
			hiddenNeurons[i] = sigmoid(hiddenIn);
		}
		for(int i=0;i<OUTPUT;i++){
			outputIn=0.0;
			for(int j=0;j<HIDDEN;j++){
				outputIn+=weightOutput[j][i]*hiddenNeurons[j];
			}
			outputNeurons[i] = sigmoid(outputIn);
		}		
	}

	private static void propagateInputsForwardTest() {
		for(int i=0;i<HIDDEN;i++){
			hiddenInTest = 0.0;
			for(int j=0;j<INPUT;j++){
				hiddenInTest+=weightsHiddenTest[j][i]*inputNeuronsTest[j];
				//uncomment below line and comment above line to run test after fresh training
				//hiddenInTest+=weightsHidden[j][i]*inputNeuronsTest[j];
			}
			hiddenNeuronsTest[i] = sigmoid(hiddenInTest);
		}
		for(int i=0;i<OUTPUT;i++){
			outputInTest=0.0;
			for(int j=0;j<HIDDEN;j++){
				outputInTest+=weightOutputTest[j][i]*hiddenNeuronsTest[j];
				//uncomment below line and comment above line to run test after fresh training
				//outputInTest+=weightOutput[j][i]*hiddenNeuronsTest[j];
			}
			outputNeuronsTest[i] = sigmoid(outputInTest);
		}
		compareOutputNeurons(outputNeuronsTest);
	}

	private static void compareOutputNeurons(double[] outputNeuronsTest) {
		if(outputNeuronsTest[0]>outputNeuronsTest[1] && outputNeuronsTest[0]>outputNeuronsTest[2])
			System.out.println("English");
		if(outputNeuronsTest[1]>outputNeuronsTest[0] && outputNeuronsTest[1]>outputNeuronsTest[2])
			System.out.println("Italian");
		if(outputNeuronsTest[2]>outputNeuronsTest[1] && outputNeuronsTest[2]>outputNeuronsTest[0])
			System.out.println("Dutch");		
	}

	private static double sigmoid(double in){
		return (1.0/(1.0+Math.exp(-1*in)));
	}

	private static void setInputAndOutputNeuronsTest(){
		inputNeuronsTest[0] = englishTest;
		inputNeuronsTest[1] = italianTest;
		inputNeuronsTest[2] = dutchTest;				
	}

	private static void setInputAndOutputNeurons() {
		if(number%3==0){
			inputNeurons[0] = englishTrain[localEnglishCount++];
			inputNeurons[1] = inputNeurons[2] =0;
			expectedNeurons[0]= 1;
			expectedNeurons[1] = 0;
			expectedNeurons[2] = 0;
		}else if(number%3==1){
			inputNeurons[0] = inputNeurons[2] =0;
			inputNeurons[1] = italianTrain[localItalianCount++];
			expectedNeurons[0] = 0;
			expectedNeurons[1]= 1;
			expectedNeurons[2] = 0;
		}else if(number%3==2){
			inputNeurons[2] = dutchTrain[localDutchCount++];
			expectedNeurons[0] = 0;
			expectedNeurons[1] = 0;
			expectedNeurons[2]= 1;
		}
	}

	private static void assignRandomWeights() {
		for(int i = 0;i<=INPUT;i++){
			for(int j=0;j<HIDDEN;j++){
				weightsHidden[i][j] = new Random().nextDouble();
			}
		}
		for(int i=0;i<=HIDDEN;i++){
			for(int j=0;j<OUTPUT;j++){
				weightOutput[i][j] = new Random().nextDouble();
			}
		}		
	}

	private static double[] getPercentOfPattern(String filename, String pattern) throws IOException{
		double[] result = new double[200];
		BufferedReader reader = new BufferedReader(new FileReader(System.getProperty("user.dir")+filename));
		String line = null;
		int wordCount = 0;
		int i = 0;
		int patternCount = 0;
		char c;
		while((line = reader.readLine())!=null){
			String[] words = line.split(" ");
			for(String word : words){
				if(pattern.equalsIgnoreCase("") && word.length()>1){
					c = word.charAt(word.length()-1);
					if(c=='a' ||c=='e' ||c=='i' ||c=='o' ||c=='u'||
							c=='A' ||c=='E' ||c=='I' ||c=='O' ||c=='U')
						patternCount++;
				}else if(pattern.equalsIgnoreCase("th")){{
					if(word.indexOf(pattern) != -1)
						patternCount++;
				}
				}else if(pattern.equalsIgnoreCase("ij")){
					if(word.indexOf(pattern) != -1)
						patternCount++;
					if(word.length()> 1 && word.charAt(word.length()-2) == 'e' && word.charAt(word.length()-1) == 'n')
						patternCount++;
				}
			}
			wordCount += line.isEmpty()?0:words.length;
			if(wordCount > 2000){
				result[i] = (float)patternCount/wordCount;
				i++;
				wordCount = 0;
				patternCount = 0;
			}
		}
		return result;		
	}

	private static double getPercentOfPatternTest(String line, String pattern) throws IOException{
		double result = 0.0;
		int wordCount = 0;
		int patternCount = 0;
		char c;
		if((line)!=null){
			String[] words = line.split(" ");
			for(String word : words){
				if(pattern.equalsIgnoreCase("") && word.length()>1){
					c = word.charAt(word.length()-1);
					if(c=='a' ||c=='e' ||c=='i' ||c=='o' ||c=='u'||
							c=='A' ||c=='E' ||c=='I' ||c=='O' ||c=='U')
						patternCount++;
				}else if(pattern.equalsIgnoreCase("th")){{
					if(word.indexOf(pattern) != -1)
						patternCount++;
				}
				}else if(pattern.equalsIgnoreCase("ij")){
					if(word.indexOf(pattern) != -1)
						patternCount++;
					if(word.length()> 1 && word.charAt(word.length()-2) == 'e' && word.charAt(word.length()-1) == 'n')
						patternCount++;
				}
			}
			wordCount += line.isEmpty()?0:words.length;
			result = (float)patternCount/wordCount;
		}
		return result;		
	}
	public static void main(String[] args) throws IOException{
		//uncomment to run trainer
		//trainData();
		//backPropLearning();
		testData();		
	}

	private static void testData() throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		String in = reader.readLine();
		englishTest = getPercentOfPatternTest(in, "th");
		dutchTest = getPercentOfPatternTest(in, "ij");
		italianTest = getPercentOfPatternTest(in, "");	
		setInputAndOutputNeuronsTest();
		propagateInputsForwardTest();
	}

	private static void trainData() throws IOException {
		englishTrain = getPercentOfPattern("/English.txt", "th");
		dutchTrain = getPercentOfPattern("/Dutch.txt", "ij");
		italianTrain = getPercentOfPattern("/Italian.txt", "");
	}
}

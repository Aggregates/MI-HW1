//**************************************
//* 
//*   Source code for Spirals Task data generation
//*	
//*	  How to execute :  
//*			Option1 :   (a) Use terminal to go to the folder. Compile the program by typing "javac GenerateSpiralsData.java"
//*					    (b) Run the program by typing "java GenerateSpiralsData"
//*			Option2 :	Use Java IDE for example Eclipse to import the source, and compile the project then execute
//*
//*
//*	  Summary of Code:
//*			There are three static functions:
//*					generateTwoSpiralsData()  --> When called, will generate original two spirals data and print on the terminal
//*					generateFourSpiralsData() --> Generates four spirals each one is out-of-phase from the other
//*					main()					 ---> Used to call either one of the above two
//**************************************





import java.util.Scanner;
import java.util.Random;
import java.math.*;
import javax.swing.JFrame;


public class Generate4SpiralsData 
{
	
	
	public static void generateTwoSpiralsData()
	{
		Scanner input = new Scanner(System.in);
		
		//number of points per spiral
		System.out.println("\nPlease enter the number of points per spiral.");
		int points = input.nextInt();
		
		//Radius
		System.out.println("\nPlease enter the radius parameter.");
		double radius = input.nextDouble();
		
		//Density
		//System.out.println("\nPlease enter the density parameter.");
		//int density = input.nextInt();
		

		System.out.println("#,x,y,class");
		
		for (int i = 0; i < points; i++){
			double phi = ((i*Math.PI)/16) ;
			double r = radius * (104 - i) / 104;
			double x = r * Math.cos(phi);
			double y = r * Math.sin(phi);
			System.out.print(x + "," + y +",RED\n");
			System.out.print(-x + "," + -y +",BLUE\n"); 
		}
		
	}
	
	public static  void generateFourSpiralsData()
	{
		Scanner input = new Scanner(System.in);
		

		System.out.println("\nPlease enter the number of pts per spiral.");
		int points = input.nextInt();
		

		System.out.println("\nPlease enter the radius parameter.");
		double radius = input.nextDouble();
		

		//System.out.println("\nPlease enter the density parameter.");
		//int density = input.nextInt();
		

		System.out.println("x,y,class");
		
		//Generate the first and second spiral using the equations in
		//Chalup Wiklendt 2007 Variations of the two-spiral task
		//Connection Science Vol. 19, No. 2, June 2007, 183-199
		for (int i = 0; i < points; i++){
			double phi = ((i*Math.PI)/16) ;
			double r = radius * (104 - i) / 104;
			double x = r * Math.cos(phi);
			double y = r * Math.sin(phi);
			System.out.print(x + "," + y +",one\n");
			System.out.print(-x + "," + -y +",two\n");
		}
		
		//Generate the third and fourth spiral with a changed phase (i+4)
		for (int i = 0; i < points; i++){
			double phi = (((i+4)*Math.PI)/16) ;
			double r = radius * (104 - i) / 104;
			double x = r * Math.cos(phi);
			double y = r * Math.sin(phi);
			System.out.print(x + "," + y +",three\n");
			System.out.print(-x + "," + -y +",four\n");
		}
		
/*		//Two more spirals with changed phase
		for (int i = 0; i < points; i++){
			double phi = (((i+10)*Math.PI)/16) ;
			double r = radius * (104 - i) / 104;
			double x = r * Math.cos(phi);
			double y = r * Math.sin(phi);
			System.out.print(x + "," + y +",THIRDBLUE\n");
			System.out.print(-x + "," + -y +",THIRDRED\n");
		}
		
		//Two more spirals with changed phase
		for (int i = 0; i < points; i++){
			double phi = (((i+14)*Math.PI)/16) ;
			double r = radius * (104 - i) / 104;
			double x = r * Math.cos(phi);
			double y = r * Math.sin(phi);
			System.out.print(x + "," + y +",FOURTHBLUE\n");
			System.out.print(-x + "," + -y +",FOURTHRED\n");
		}
 */
	}
	
	
	
	
	public static void main(String[] args) 
	{
		
		//generateTwoSpiralsData();
		generateFourSpiralsData();
	}
	
}


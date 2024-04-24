#include <iostream>
#include <stdlib.h>

using namespace std;

typedef struct Node *NodePointer;

void push (NodePointer &Head, int Number);
void print_list(NodePointer &Head);
struct Node{
	int Data;
	NodePointer Link;
}; 

int main(){
	 int number; 
	 NodePointer Head;
	 NodePointer Second;
	 NodePointer Third;
	 
	 Head = new Node();
	 Second = new Node();
	 Third = new Node();
	 
	 Head->Data = 38;
	 Head->Link = Second;
	 Second->Data = 88;
	 Second->Link = Third;
	 Third->Data = 49;
	 Third->Link = NULL;
	 
	 print_list (Head);
	 cout<<"\nEnter a number: ";
	 cin>>number;
	 push(Head, number);
	 print_list (Head);
}

void print_list (NodePointer &Head){
	NodePointer current;
	current = Head;
	while(current != NULL){
		cout<<"current -> Data"<<" ";
		current = current->Link;
	}
}
void push (NodePointer &Head, int Number){
	NodePointer NewNode;
	NewNode = new Node();
	NewNode ->Data = Number;
	Head = NewNode;
	cout<<"\nNew node["<<NewNode->Data<<"]pushed to linked list"; 
}	
void pop(NodePointer &Head);

void pop(NodePointer &Head){
	NodePointer temp;
	temp = Head;
	Head = Head ->Link;
	int num = temp ->Data;
	delete temp;
	cout<<"\nRemove["<<num<<"] "<<endl;
}

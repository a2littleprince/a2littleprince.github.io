var json_data=  {
	"question": "Why do I have to pay a service fee?",
	"answer" : "A service fee is only charged for repairs to devices that are no longer under war-ran-ty. Business customers are not charged a service fee in accord with the terms of their con-tract."
  };
var faq=[
{
"question": "Why do I have to pay a service fee?",
"answer" : "A service fee is only charged for repairs to devices that are no longer under war-ran-ty. Business customers are not charged a service fee in accord with the terms of their con-tract."
},

{
"question": "What is the bond for?",
"answer" : "The bond is to cover any damage done to the courtesy phone and/or charger. The bond will be refunded upon the safe and undamaged return of the phone and charger."
},

{
"question": "Do I need a charger with my courtesy phone?",
"answer" : "No, a charger is optional. You can add one with the 'Add charger' button. If you don't want one but added one by accident, you can remove it by clicking on the 'Remove charger' button."
},

{
"question": " Why isn't my phone under warranty?",
"answer" : " The length of your phone's warranty depends on the warranty package you chose upon purchase. The standard is 24 months and is calculated from its purchase date."
},

{
"question": " How long will my repair take?",
"answer" : " Depends on your phone broken status. It takes normally 5 to 7 working days."
},

{
"question": " How do you protect the private information in my phone?",
"answer" : " We comply with all relevant laws regarding privacy and client confidentiality."
},

{
"question": " Where do you get your replacement parts?",
"answer" : " We will send you a quote of all possible vendors who supply replacement parts for your references and your choice."
},

{
"question": " What happens if my phone is further damage after leaving it with you?",
"answer" : " We make sure that it never happens."
},

{
"question": " What kind of warranty do you offer and what does it cover?",
"answer" : "1 month is the average warranty period. These warranties covers parts and service only."
},

{
"question": " What does the repair estimate include?",
"answer" : " The repair price estimate includes both replacement parts and labor."
}   
];


function renderHTML(data) {     //Build an html string which will be rendered on browser as an html-formatted element    
let htmlString = "";       //Retrieve question and relevant answer   
for (i = 0; i < data.length; i++) {       //Get question      
 htmlString += "<h4>" + data[i].question + "</h4>";      
 //Get answer    
 htmlString += "<p>" + data[i].answer + "</p><br>";     }        
//Render the whole htmlString to web page  
document.getElementById("questions").innerHTML = htmlString;  
}  
renderHTML(faq);  
var keyword = document.getElementById('keyword');
        // console.log(keyword);
        var sear = document.getElementById('search');
        var content = document.getElementById("questions");
        keyword.onblur = function() {
            sear.onclick = function() {
                let textstr = '';
                for (let i = 0; i < faq.length; i++) {
                    var res = faq[i]['question'].indexOf(keyword.value);
                    console.log(faq[i]['question']);
                    console.log(res);
                    if (res != -1) {
                        textstr += '<p>' + faq[i]['answer'] + '</p>';
                    }
                }
                if (!keyword.value || !textstr) {
                    content.innerHTML = '<p>There  is no results</p>'
                } else {
                    content.innerHTML = textstr;
                }

          }
        }
//1 : declare JSON array
var faq2=
[
{
"question": "Why do I have to pay a service fee?",
"answer" : "A service fee is only charged for repairs to devices that are no longer under war-ran-ty. Business customers are not charged a service fee in accord with the terms of their con-tract."
},

{
"question": "What is the bond for?",
"answer" : "The bond is to cover any damage done to the courtesy phone and/or charger. The bond will be refunded upon the safe and undamaged return of the phone and charger."
},

{
"question": "Do I need a charger with my courtesy phone?",
"answer" : "No, a charger is optional. You can add one with the 'Add charger' button. If you don't want one but added one by accident, you can remove it by clicking on the 'Remove charger' button."
},

{
"question": " Why isn't my phone under warranty?",
"answer" : " The length of your phone's warranty depends on the warranty package you chose upon purchase. The standard is 24 months and is calculated from its purchase date."
},

{
"question": " How long will my repair take?",
"answer" : " Depends on your phone broken status. It takes normally 5 to 7 working days."
},

{
"question": " How do you protect the private information in my phone?",
"answer" : " We comply with all relevant laws regarding privacy and client confidentiality."
},

{
"question": " Where do you get your replacement parts?",
"answer" : " We will send you a quote of all possible vendors who supply replacement parts for your references and your choice."
},

{
"question": " What happens if my phone is further damage after leaving it with you?",
"answer" : " We make sure that it never happens."
},

{
"question": " What kind of warranty do you offer and what does it cover?",
"answer" : "1 month is the average warranty period. These warranties covers parts and service only."
},

{
"question": " What does the repair estimate include?",
"answer" : " The repair price estimate includes both replacement parts and labor."
}   
]

//2 create a function to render/read the content of JSON
function renderHTML2(data) {    
//Build an html string which will be rendered on browser as an html-formatted element   
let htmlString = "";       //Retrieve question and relevant answer     
for (i = 0; i < data.length; i++) {   
  //Get question      
  htmlString += "<h4>" + data[i].question + "</h4>";       
  //Get answer   
  htmlString += "<p>" + data[i].answer + "</p><br>";     
}          
//Render the whole htmlString to web page 
document.getElementById("faq").innerHTML = htmlString;   }  

//3 Call the render function
renderHTML2(faq2); //call the function
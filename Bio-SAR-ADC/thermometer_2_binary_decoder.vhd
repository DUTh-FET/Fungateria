----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 02/04/2025 11:12:07 AM
-- Design Name: 
-- Module Name: thermometer_2_binary_decoder - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;


entity thermometer_2_binary_decoder is
  Port (
    T : in STD_LOGIC_VECTOR(3 downto 0); -- 4-bit thermometer code
    B: out STD_LOGIC_VECTOR(1 downto 0); -- 2-bit binary output
    overflow: out STD_LOGIC -- overflow flag
   );
end thermometer_2_binary_decoder;

architecture Behavioral of thermometer_2_binary_decoder is

begin
    process (T)
    begin
        case T is
            when "0000" => B <= "00"; overflow <= '0'; -- No threshold exceeded
            when "0001" => B <= "01"; overflow <= '0';
            when "0011" => B <= "10"; overflow <= '0';
            when "0111" => B <= "11"; overflow <= '0';
            when "1111" => B <= "11"; overflow <= '1'; -- All thresholds exceeded
                
                
        
            when others => B <= "11"; overflow <= '1'; -- All thresholds exceeded
        end case;
    end process;


end Behavioral;

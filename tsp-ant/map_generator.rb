class City
  attr_accessor :x,:y;

  def initialize
    @x = rand(100)
    @y = rand(100)
  end

end


class MapGenerator
  require 'matrix'
  #require 'math'
  MaxCity_CONST = ARGV[0].to_i # max number of cities
    
  def main()
      @map = generate_cities(MaxCity_CONST) # Plot Cities 
      @distances = calculate_distances(@map)
      generate_output(@distances)
  end
 
  #Cities -> integer
  def generate_cities(cities)
      map =  Array.new

      (0..cities-1).each do 
          map.push(City.new)
      end
      return map
  end

  #map -> Array of City
  def calculate_distances(map)
    distances = Array.new(map.size) { Array.new(map.size) }
    (0..map.size-1).each do |i|
      (0..map.size-1).each do |j|
        #distance between two points
        if j == i
          distances[i][j] = 0
          distances[j][i] = 0
        else
          dis = Math.hypot((map[i].x - map[j].x),(map[i].y - map[j].y))
          distances[i][j] = dis
          distances[j][i] = dis
        end
      end       
    end

    return distances
  end

  #distances -> Matrix of distances
  def generate_output(distances)
      if File.exist?("map.txt")
       puts "The file already exists. Rebuilding it..."
       system("rm map.txt")
      end
        out = File.open("map.txt","w")

      (0..distances.size-1).each do |i|
        (0..distances.size-1).each do |j|
            out.puts("#{i} #{j} #{distances[i][j].round(4)}")
        end
      end
  end
  
  def initialize
    main()
  end

end


MapGenerator.new
